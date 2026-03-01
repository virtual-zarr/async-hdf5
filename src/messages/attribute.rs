use std::sync::Arc;

use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::Result;
use crate::heap;
use crate::messages::dataspace::DataspaceMessage;
use crate::messages::datatype::{ByteOrder, DataType, StringPadding};
use crate::reader::AsyncFileReader;

/// A decoded HDF5 attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    /// Signed 8-bit integer (scalar or array).
    I8(Vec<i8>),
    /// Signed 16-bit integer (scalar or array).
    I16(Vec<i16>),
    /// Signed 32-bit integer (scalar or array).
    I32(Vec<i32>),
    /// Signed 64-bit integer (scalar or array).
    I64(Vec<i64>),
    /// Unsigned 8-bit integer (scalar or array).
    U8(Vec<u8>),
    /// Unsigned 16-bit integer (scalar or array).
    U16(Vec<u16>),
    /// Unsigned 32-bit integer (scalar or array).
    U32(Vec<u32>),
    /// Unsigned 64-bit integer (scalar or array).
    U64(Vec<u64>),
    /// 32-bit float (scalar or array).
    F32(Vec<f32>),
    /// 64-bit float (scalar or array).
    F64(Vec<f64>),
    /// Fixed-length string.
    String(String),
    /// Raw bytes (for types we don't decode).
    Raw(Vec<u8>),
}

impl AttributeValue {
    /// Returns true if this is a scalar (single-element) value.
    pub fn is_scalar(&self) -> bool {
        match self {
            AttributeValue::I8(v) => v.len() == 1,
            AttributeValue::I16(v) => v.len() == 1,
            AttributeValue::I32(v) => v.len() == 1,
            AttributeValue::I64(v) => v.len() == 1,
            AttributeValue::U8(v) => v.len() == 1,
            AttributeValue::U16(v) => v.len() == 1,
            AttributeValue::U32(v) => v.len() == 1,
            AttributeValue::U64(v) => v.len() == 1,
            AttributeValue::F32(v) => v.len() == 1,
            AttributeValue::F64(v) => v.len() == 1,
            AttributeValue::String(_) => true,
            AttributeValue::Raw(_) => true,
        }
    }

    /// Try to get a scalar i32 value.
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            AttributeValue::I32(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        }
    }

    /// Try to get a scalar i64 value.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            AttributeValue::I64(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        }
    }

    /// Try to get a scalar f32 value.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            AttributeValue::F32(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        }
    }

    /// Try to get a scalar f64 value.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            AttributeValue::F64(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        }
    }

    /// Try to get a string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            AttributeValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// A fully resolved HDF5 attribute: name + decoded value.
#[derive(Debug, Clone)]
pub struct Attribute {
    /// Attribute name.
    pub name: String,
    /// Decoded value.
    pub value: AttributeValue,
}

/// An HDF5 attribute (name-value pair attached to a group or dataset).
///
/// Message type 0x000C.
#[derive(Debug, Clone)]
pub struct AttributeMessage {
    /// Attribute name.
    pub name: String,
    /// Data type of the attribute value.
    pub dtype: DataType,
    /// Dataspace (dimensionality) of the attribute.
    pub dataspace: DataspaceMessage,
    /// Raw value bytes.
    pub raw_value: Bytes,
}

impl AttributeMessage {
    /// Decode the raw value bytes into a typed `AttributeValue`.
    pub fn decode(&self) -> AttributeValue {
        let n = num_elements(&self.dataspace.dimensions) as usize;
        let raw = &self.raw_value;

        match &self.dtype {
            DataType::FixedPoint {
                size,
                signed,
                byte_order,
                ..
            } => decode_fixed_point(raw, *size, *signed, *byte_order, n),

            DataType::FloatingPoint {
                size, byte_order, ..
            } => decode_floating_point(raw, *size, *byte_order, n),

            DataType::String {
                size, padding, ..
            } => {
                let s = if *size == 0 {
                    // Zero-size means the string is empty or this is a vlen string
                    String::new()
                } else {
                    let end = (*size as usize).min(raw.len());
                    let bytes = &raw[..end];
                    let s = String::from_utf8_lossy(bytes);
                    match padding {
                        StringPadding::NullTerminate => {
                            s.split('\0').next().unwrap_or("").to_string()
                        }
                        StringPadding::NullPad => s.trim_end_matches('\0').to_string(),
                        StringPadding::SpacePad => s.trim_end().to_string(),
                    }
                };
                AttributeValue::String(s)
            }

            // Enum with 1-byte base → treat as bool (h5py convention: FALSE=0, TRUE=1)
            DataType::Enum { base_type, .. } => {
                // Decode as the base type
                match base_type.as_ref() {
                    DataType::FixedPoint {
                        size,
                        signed,
                        byte_order,
                        ..
                    } => decode_fixed_point(raw, *size, *signed, *byte_order, n),
                    _ => AttributeValue::Raw(raw.to_vec()),
                }
            }

            _ => AttributeValue::Raw(raw.to_vec()),
        }
    }

    /// Convert this message to a resolved `Attribute` with decoded value.
    pub fn to_attribute(&self) -> Attribute {
        Attribute {
            name: self.name.clone(),
            value: self.decode(),
        }
    }

    /// Decode the raw value, resolving variable-length data via the global heap.
    ///
    /// This handles the common case of vlen strings written by h5py (the default).
    /// Falls back to `decode()` for non-vlen types.
    pub async fn decode_with_reader(
        &self,
        reader: &Arc<dyn AsyncFileReader>,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<AttributeValue> {
        match &self.dtype {
            DataType::VarLen { is_string: true, .. } => {
                // Vlen string: raw_value = length(4) + collection_addr(size_of_offsets) + object_index(4)
                let raw = &self.raw_value;
                if raw.len() < 4 + size_of_offsets as usize + 4 {
                    return Ok(AttributeValue::String(String::new()));
                }
                let mut r = HDF5Reader::with_sizes(
                    self.raw_value.clone(),
                    size_of_offsets,
                    size_of_lengths,
                );
                let _seq_len = r.read_u32()?;
                let collection_addr = r.read_offset()?;
                let object_index = r.read_u32()?;

                // Undefined address means empty/null
                if HDF5Reader::is_undef_addr(collection_addr, size_of_offsets) {
                    return Ok(AttributeValue::String(String::new()));
                }

                let obj_data = heap::global::read_global_heap_object(
                    reader,
                    collection_addr,
                    object_index,
                    size_of_offsets,
                    size_of_lengths,
                )
                .await?;

                let s = String::from_utf8_lossy(&obj_data);
                // Vlen strings are null-terminated by convention
                let s = s.split('\0').next().unwrap_or("").to_string();
                Ok(AttributeValue::String(s))
            }
            _ => Ok(self.decode()),
        }
    }

    /// Async version of `to_attribute` that resolves vlen data.
    pub async fn to_attribute_resolved(
        &self,
        reader: &Arc<dyn AsyncFileReader>,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Attribute> {
        let value = self
            .decode_with_reader(reader, size_of_offsets, size_of_lengths)
            .await?;
        Ok(Attribute {
            name: self.name.clone(),
            value,
        })
    }

    /// Parse from the raw message bytes.
    pub fn parse(data: &Bytes, size_of_offsets: u8, size_of_lengths: u8) -> Result<Self> {
        let mut r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);

        let version = r.read_u8()?;

        match version {
            1 => Self::parse_v1(&mut r, data, size_of_offsets, size_of_lengths),
            2 => Self::parse_v2(&mut r, data, size_of_offsets, size_of_lengths),
            3 => Self::parse_v3(&mut r, data, size_of_offsets, size_of_lengths),
            _ => {
                // Best effort — return minimal
                Ok(Self {
                    name: String::new(),
                    dtype: DataType::Opaque {
                        size: 0,
                        tag: String::new(),
                    },
                    dataspace: DataspaceMessage {
                        rank: 0,
                        dimensions: vec![],
                        max_dimensions: None,
                    },
                    raw_value: Bytes::new(),
                })
            }
        }
    }

    fn parse_v1(
        r: &mut HDF5Reader,
        data: &Bytes,
        _size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        r.skip(1); // reserved
        let name_size = r.read_u16()? as usize;
        let datatype_size = r.read_u16()? as usize;
        let dataspace_size = r.read_u16()? as usize;

        // Name (padded to 8-byte boundary)
        let name_bytes = r.read_bytes(name_size)?;
        let name = String::from_utf8_lossy(&name_bytes)
            .trim_end_matches('\0')
            .to_string();
        let name_pad = (8 - (name_size % 8)) % 8;
        r.skip(name_pad as u64);

        // Datatype
        let dt_start = r.position() as usize;
        let dt_bytes = data.slice(dt_start..dt_start + datatype_size);
        let dtype = DataType::parse(&dt_bytes)?;
        r.skip(datatype_size as u64);
        let dt_pad = (8 - (datatype_size % 8)) % 8;
        r.skip(dt_pad as u64);

        // Dataspace
        let ds_start = r.position() as usize;
        let ds_bytes = data.slice(ds_start..ds_start + dataspace_size);
        let dataspace = DataspaceMessage::parse(&ds_bytes, size_of_lengths)?;
        r.skip(dataspace_size as u64);
        let ds_pad = (8 - (dataspace_size % 8)) % 8;
        r.skip(ds_pad as u64);

        // Value
        let raw_value = extract_raw_value(r, data, &dataspace, &dtype);

        Ok(Self {
            name,
            dtype,
            dataspace,
            raw_value,
        })
    }

    fn parse_v2(
        r: &mut HDF5Reader,
        data: &Bytes,
        _size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        // v2: same as v1 but no padding on name/dt/ds
        let _flags = r.read_u8()?;
        let name_size = r.read_u16()? as usize;
        let datatype_size = r.read_u16()? as usize;
        let dataspace_size = r.read_u16()? as usize;

        let name_bytes = r.read_bytes(name_size)?;
        let name = String::from_utf8_lossy(&name_bytes)
            .trim_end_matches('\0')
            .to_string();

        let dt_start = r.position() as usize;
        let dt_bytes = data.slice(dt_start..dt_start + datatype_size);
        let dtype = DataType::parse(&dt_bytes)?;
        r.skip(datatype_size as u64);

        let ds_start = r.position() as usize;
        let ds_bytes = data.slice(ds_start..ds_start + dataspace_size);
        let dataspace = DataspaceMessage::parse(&ds_bytes, size_of_lengths)?;
        r.skip(dataspace_size as u64);

        let raw_value = extract_raw_value(r, data, &dataspace, &dtype);

        Ok(Self {
            name,
            dtype,
            dataspace,
            raw_value,
        })
    }

    fn parse_v3(
        r: &mut HDF5Reader,
        data: &Bytes,
        _size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        // v3: same as v2 but with creation order
        let flags = r.read_u8()?;
        let name_size = r.read_u16()? as usize;
        let datatype_size = r.read_u16()? as usize;
        let dataspace_size = r.read_u16()? as usize;

        let _charset = if flags & 0x10 != 0 {
            r.read_u8()?
        } else {
            0
        };

        let name_bytes = r.read_bytes(name_size)?;
        let name = String::from_utf8_lossy(&name_bytes)
            .trim_end_matches('\0')
            .to_string();

        let dt_start = r.position() as usize;
        let dt_bytes = data.slice(dt_start..dt_start + datatype_size);
        let dtype = DataType::parse(&dt_bytes)?;
        r.skip(datatype_size as u64);

        let ds_start = r.position() as usize;
        let ds_bytes = data.slice(ds_start..ds_start + dataspace_size);
        let dataspace = DataspaceMessage::parse(&ds_bytes, size_of_lengths)?;
        r.skip(dataspace_size as u64);

        let raw_value = extract_raw_value(r, data, &dataspace, &dtype);

        Ok(Self {
            name,
            dtype,
            dataspace,
            raw_value,
        })
    }
}

/// Compute the total number of elements from dataspace dimensions using
/// saturating arithmetic to avoid overflow panics on malformed data.
fn num_elements(dimensions: &[u64]) -> u64 {
    dimensions
        .iter()
        .copied()
        .fold(1u64, |acc, d| acc.saturating_mul(d))
        .max(1)
}

/// Extract the raw value bytes at the current reader position, using the
/// dataspace dimensions and dtype size to compute how many bytes to read.
fn extract_raw_value(
    r: &HDF5Reader,
    data: &Bytes,
    dataspace: &DataspaceMessage,
    dtype: &DataType,
) -> Bytes {
    let n = num_elements(&dataspace.dimensions);
    let value_size = n.saturating_mul(dtype.size() as u64) as usize;
    let val_start = r.position() as usize;
    if val_start + value_size <= data.len() {
        data.slice(val_start..val_start + value_size)
    } else {
        Bytes::new()
    }
}

/// Decode fixed-point (integer) raw bytes into an `AttributeValue`.
fn decode_fixed_point(
    raw: &[u8],
    size: u32,
    signed: bool,
    byte_order: ByteOrder,
    n: usize,
) -> AttributeValue {
    let is_le = matches!(byte_order, ByteOrder::LittleEndian);
    match (size, signed) {
        (1, true) => {
            let vals: Vec<i8> = raw.iter().take(n).map(|&b| b as i8).collect();
            AttributeValue::I8(vals)
        }
        (1, false) => {
            let vals: Vec<u8> = raw.iter().take(n).copied().collect();
            AttributeValue::U8(vals)
        }
        (2, true) => {
            let vals: Vec<i16> = raw
                .chunks_exact(2)
                .take(n)
                .map(|c| {
                    if is_le {
                        i16::from_le_bytes([c[0], c[1]])
                    } else {
                        i16::from_be_bytes([c[0], c[1]])
                    }
                })
                .collect();
            AttributeValue::I16(vals)
        }
        (2, false) => {
            let vals: Vec<u16> = raw
                .chunks_exact(2)
                .take(n)
                .map(|c| {
                    if is_le {
                        u16::from_le_bytes([c[0], c[1]])
                    } else {
                        u16::from_be_bytes([c[0], c[1]])
                    }
                })
                .collect();
            AttributeValue::U16(vals)
        }
        (4, true) => {
            let vals: Vec<i32> = raw
                .chunks_exact(4)
                .take(n)
                .map(|c| {
                    if is_le {
                        i32::from_le_bytes([c[0], c[1], c[2], c[3]])
                    } else {
                        i32::from_be_bytes([c[0], c[1], c[2], c[3]])
                    }
                })
                .collect();
            AttributeValue::I32(vals)
        }
        (4, false) => {
            let vals: Vec<u32> = raw
                .chunks_exact(4)
                .take(n)
                .map(|c| {
                    if is_le {
                        u32::from_le_bytes([c[0], c[1], c[2], c[3]])
                    } else {
                        u32::from_be_bytes([c[0], c[1], c[2], c[3]])
                    }
                })
                .collect();
            AttributeValue::U32(vals)
        }
        (8, true) => {
            let vals: Vec<i64> = raw
                .chunks_exact(8)
                .take(n)
                .map(|c| {
                    if is_le {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                    } else {
                        i64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                    }
                })
                .collect();
            AttributeValue::I64(vals)
        }
        (8, false) => {
            let vals: Vec<u64> = raw
                .chunks_exact(8)
                .take(n)
                .map(|c| {
                    if is_le {
                        u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                    } else {
                        u64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                    }
                })
                .collect();
            AttributeValue::U64(vals)
        }
        _ => AttributeValue::Raw(raw.to_vec()),
    }
}

/// Decode floating-point raw bytes into an `AttributeValue`.
fn decode_floating_point(
    raw: &[u8],
    size: u32,
    byte_order: ByteOrder,
    n: usize,
) -> AttributeValue {
    let is_le = matches!(byte_order, ByteOrder::LittleEndian);
    match size {
        4 => {
            let vals: Vec<f32> = raw
                .chunks_exact(4)
                .take(n)
                .map(|c| {
                    if is_le {
                        f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                    } else {
                        f32::from_be_bytes([c[0], c[1], c[2], c[3]])
                    }
                })
                .collect();
            AttributeValue::F32(vals)
        }
        8 => {
            let vals: Vec<f64> = raw
                .chunks_exact(8)
                .take(n)
                .map(|c| {
                    if is_le {
                        f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                    } else {
                        f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                    }
                })
                .collect();
            AttributeValue::F64(vals)
        }
        _ => AttributeValue::Raw(raw.to_vec()),
    }
}
