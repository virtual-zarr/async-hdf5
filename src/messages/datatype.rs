use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};

/// Byte order for multi-byte data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// Little-endian byte order.
    LittleEndian,
    /// Big-endian byte order.
    BigEndian,
    /// VAX-endian (rare, HDF5 legacy).
    Vax,
    /// Not applicable (e.g., single-byte types).
    NotApplicable,
}

/// String padding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringPadding {
    /// Null-terminated string.
    NullTerminate,
    /// Null-padded string.
    NullPad,
    /// Space-padded string.
    SpacePad,
}

/// String character set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Charset {
    /// ASCII character set.
    Ascii,
    /// UTF-8 character set.
    Utf8,
}

/// A field within a compound data type.
#[derive(Debug, Clone)]
pub struct CompoundField {
    /// Field name.
    pub name: String,
    /// Byte offset of the field within the compound type.
    pub byte_offset: u32,
    /// Data type of the field.
    pub dtype: DataType,
}

/// HDF5 data type descriptor.
///
/// Parsed from the Datatype message (0x0003) in object headers.
#[derive(Debug, Clone)]
pub enum DataType {
    /// Fixed-point integers (signed/unsigned, various sizes).
    FixedPoint {
        /// Total size in bytes.
        size: u32,
        /// Whether the integer is signed.
        signed: bool,
        /// Byte order.
        byte_order: ByteOrder,
        /// Bit offset of the first significant bit.
        bit_offset: u16,
        /// Number of significant bits.
        bit_precision: u16,
    },
    /// IEEE 754 floating-point.
    FloatingPoint {
        /// Total size in bytes.
        size: u32,
        /// Byte order.
        byte_order: ByteOrder,
        /// Bit offset of the first significant bit.
        bit_offset: u16,
        /// Number of significant bits.
        bit_precision: u16,
        /// Bit position of the exponent field.
        exponent_location: u8,
        /// Size of the exponent field in bits.
        exponent_size: u8,
        /// Bit position of the mantissa field.
        mantissa_location: u8,
        /// Size of the mantissa field in bits.
        mantissa_size: u8,
        /// Exponent bias.
        exponent_bias: u32,
    },
    /// Fixed-length or variable-length strings.
    String {
        /// Total size in bytes (0 for variable-length).
        size: u32,
        /// Padding type.
        padding: StringPadding,
        /// Character set.
        charset: Charset,
    },
    /// Compound types (e.g., CFloat32 = {r: float32, i: float32}).
    Compound {
        /// Total size in bytes.
        size: u32,
        /// Member fields.
        fields: Vec<CompoundField>,
    },
    /// Enumerated types.
    Enum {
        /// Total size in bytes.
        size: u32,
        /// Underlying integer type.
        base_type: Box<DataType>,
        /// Enum member names and their raw values.
        members: Vec<(String, Vec<u8>)>,
    },
    /// Variable-length sequences or strings.
    VarLen {
        /// Base element type (for sequences) or u8 (for strings).
        base_type: Box<DataType>,
        /// True if this is a variable-length string (class bits type=1).
        is_string: bool,
    },
    /// Fixed-size array of another type.
    Array {
        /// Element type.
        base_type: Box<DataType>,
        /// Array dimensions.
        dimensions: Vec<u32>,
    },
    /// Opaque data.
    Opaque {
        /// Total size in bytes.
        size: u32,
        /// ASCII tag describing the opaque type.
        tag: String,
    },
    /// Bitfield.
    Bitfield {
        /// Total size in bytes.
        size: u32,
        /// Byte order.
        byte_order: ByteOrder,
        /// Bit offset of the first significant bit.
        bit_offset: u16,
        /// Number of significant bits.
        bit_precision: u16,
    },
    /// Reference type.
    Reference {
        /// Total size in bytes.
        size: u32,
        /// Reference type (0 = object, 1 = region).
        ref_type: u8,
    },
}

impl DataType {
    /// Element size in bytes.
    pub fn size(&self) -> u32 {
        match self {
            DataType::FixedPoint { size, .. } => *size,
            DataType::FloatingPoint { size, .. } => *size,
            DataType::String { size, .. } => *size,
            DataType::Compound { size, .. } => *size,
            DataType::Enum { size, .. } => *size,
            DataType::VarLen { .. } => 16, // HDF5 vlen: {size, pointer}
            DataType::Array { base_type, dimensions } => {
                base_type.size() * dimensions.iter().product::<u32>()
            }
            DataType::Opaque { size, .. } => *size,
            DataType::Bitfield { size, .. } => *size,
            DataType::Reference { size, .. } => *size,
        }
    }

    /// Parse a datatype message from raw bytes.
    pub fn parse(data: &Bytes) -> Result<Self> {
        let mut r = HDF5Reader::new(data.clone());
        Self::parse_from_reader(&mut r)
    }

    /// Parse from an HDF5Reader (allows recursive parsing for compound types).
    pub(crate) fn parse_from_reader(r: &mut HDF5Reader) -> Result<Self> {
        let class_and_version = r.read_u8()?;
        let class = class_and_version & 0x0F;
        let version = (class_and_version >> 4) & 0x0F;

        // 3 bytes of class bit field
        let bf0 = r.read_u8()?;
        let bf1 = r.read_u8()?;
        let bf2 = r.read_u8()?;
        let class_bits = (bf2 as u32) << 16 | (bf1 as u32) << 8 | (bf0 as u32);

        let size = r.read_u32()?;

        match class {
            0 => Self::parse_fixed_point(r, class_bits, size),
            1 => Self::parse_floating_point(r, class_bits, size),
            3 => Self::parse_string(class_bits, size),
            4 => Self::parse_bitfield(r, class_bits, size),
            5 => Self::parse_opaque(r, class_bits, size),
            6 => Self::parse_compound(r, class_bits, size, version),
            7 => Ok(DataType::Reference {
                size,
                ref_type: (class_bits & 0x0F) as u8,
            }),
            8 => Self::parse_enum(r, class_bits, size),
            9 => Self::parse_varlen(r, class_bits, size),
            10 => Self::parse_array(r, class_bits, size),
            _ => Err(HDF5Error::UnsupportedDatatypeClass(class)),
        }
    }

    fn parse_fixed_point(r: &mut HDF5Reader, class_bits: u32, size: u32) -> Result<Self> {
        let byte_order = match class_bits & 0x01 {
            0 => ByteOrder::LittleEndian,
            1 => ByteOrder::BigEndian,
            _ => unreachable!(),
        };
        let signed = (class_bits >> 3) & 0x01 == 1;

        let bit_offset = r.read_u16()?;
        let bit_precision = r.read_u16()?;

        Ok(DataType::FixedPoint {
            size,
            signed,
            byte_order,
            bit_offset,
            bit_precision,
        })
    }

    fn parse_floating_point(r: &mut HDF5Reader, class_bits: u32, size: u32) -> Result<Self> {
        let byte_order = match (class_bits & 0x01, (class_bits >> 6) & 0x01) {
            (0, 0) => ByteOrder::LittleEndian,
            (1, 0) => ByteOrder::BigEndian,
            (0, 1) => ByteOrder::Vax,
            _ => ByteOrder::NotApplicable,
        };

        let bit_offset = r.read_u16()?;
        let bit_precision = r.read_u16()?;
        let exponent_location = r.read_u8()?;
        let exponent_size = r.read_u8()?;
        let mantissa_location = r.read_u8()?;
        let mantissa_size = r.read_u8()?;
        let exponent_bias = r.read_u32()?;

        Ok(DataType::FloatingPoint {
            size,
            byte_order,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        })
    }

    fn parse_string(class_bits: u32, size: u32) -> Result<Self> {
        let padding = match class_bits & 0x0F {
            0 => StringPadding::NullTerminate,
            1 => StringPadding::NullPad,
            2 => StringPadding::SpacePad,
            _ => StringPadding::NullTerminate,
        };
        let charset = match (class_bits >> 4) & 0x0F {
            0 => Charset::Ascii,
            1 => Charset::Utf8,
            _ => Charset::Ascii,
        };

        Ok(DataType::String {
            size,
            padding,
            charset,
        })
    }

    fn parse_bitfield(r: &mut HDF5Reader, class_bits: u32, size: u32) -> Result<Self> {
        let byte_order = match class_bits & 0x01 {
            0 => ByteOrder::LittleEndian,
            _ => ByteOrder::BigEndian,
        };
        let bit_offset = r.read_u16()?;
        let bit_precision = r.read_u16()?;

        Ok(DataType::Bitfield {
            size,
            byte_order,
            bit_offset,
            bit_precision,
        })
    }

    fn parse_opaque(r: &mut HDF5Reader, class_bits: u32, size: u32) -> Result<Self> {
        let tag_len = (class_bits & 0xFF) as usize;
        let tag_bytes = r.read_bytes(tag_len)?;
        let tag = String::from_utf8_lossy(&tag_bytes).trim_end_matches('\0').to_string();
        // Pad to multiple of 8
        let pad = (8 - (tag_len % 8)) % 8;
        r.skip(pad as u64);

        Ok(DataType::Opaque { size, tag })
    }

    fn parse_compound(
        r: &mut HDF5Reader,
        class_bits: u32,
        size: u32,
        version: u8,
    ) -> Result<Self> {
        let num_members = (class_bits & 0xFFFF) as usize;
        let mut fields = Vec::with_capacity(num_members);

        for _ in 0..num_members {
            // Read null-terminated name
            let mut name_bytes = Vec::new();
            loop {
                let b = r.read_u8()?;
                if b == 0 {
                    break;
                }
                name_bytes.push(b);
            }
            let name = String::from_utf8_lossy(&name_bytes).to_string();

            // Version 1 & 2: pad name to multiple of 8 bytes (including null terminator)
            if version < 3 {
                let name_total = name_bytes.len() + 1; // including null
                let pad = (8 - (name_total % 8)) % 8;
                r.skip(pad as u64);
            }

            // Byte offset of member
            let byte_offset = if version < 3 {
                // v1/v2: 4-byte offset + dimensionality info
                let offset = r.read_u32()?;
                // v1: dimensionality (1 byte) + reserved (3 bytes) + perm (4 bytes) + reserved (4 bytes) + dims (4*4 bytes)
                // v2: no dimensionality
                if version == 1 {
                    let ndims = r.read_u8()?;
                    r.skip(3 + 4 + 4); // reserved + permutation + reserved
                    r.skip(ndims as u64 * 4); // dimension sizes
                }
                offset
            } else {
                // v3: variable-size offset based on type size
                if size <= 0xFF {
                    r.read_u8()? as u32
                } else if size <= 0xFFFF {
                    r.read_u16()? as u32
                } else {
                    r.read_u32()?
                }
            };

            // Recursively parse member datatype
            let dtype = DataType::parse_from_reader(r)?;

            fields.push(CompoundField {
                name,
                byte_offset,
                dtype,
            });
        }

        Ok(DataType::Compound { size, fields })
    }

    fn parse_enum(r: &mut HDF5Reader, class_bits: u32, size: u32) -> Result<Self> {
        let num_members = (class_bits & 0xFFFF) as usize;
        let base_type = Box::new(DataType::parse_from_reader(r)?);

        let mut names = Vec::with_capacity(num_members);
        for _ in 0..num_members {
            let mut name_bytes = Vec::new();
            loop {
                let b = r.read_u8()?;
                if b == 0 {
                    break;
                }
                name_bytes.push(b);
            }
            names.push(String::from_utf8_lossy(&name_bytes).to_string());
        }

        let member_size = base_type.size() as usize;
        let mut members = Vec::with_capacity(num_members);
        for name in names {
            let value = r.read_bytes(member_size)?;
            members.push((name, value));
        }

        Ok(DataType::Enum {
            size,
            base_type,
            members,
        })
    }

    fn parse_varlen(r: &mut HDF5Reader, class_bits: u32, _size: u32) -> Result<Self> {
        let is_string = (class_bits & 0x0F) == 1;
        let base_type = Box::new(DataType::parse_from_reader(r)?);
        Ok(DataType::VarLen {
            base_type,
            is_string,
        })
    }

    fn parse_array(r: &mut HDF5Reader, _class_bits: u32, _size: u32) -> Result<Self> {
        let ndims = r.read_u8()?;
        // v3: no reserved bytes. v2: 3 reserved bytes.
        // We try the v3 path — if needed, we can add version awareness.

        let mut dimensions = Vec::with_capacity(ndims as usize);
        for _ in 0..ndims {
            dimensions.push(r.read_u32()?);
        }

        let base_type = Box::new(DataType::parse_from_reader(r)?);
        Ok(DataType::Array {
            base_type,
            dimensions,
        })
    }
}
