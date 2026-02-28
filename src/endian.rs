use byteorder::{LittleEndian, ReadBytesExt};
use bytes::Bytes;
use std::io::{Cursor, Read};

use crate::error::{HDF5Error, Result};

/// Undefined address sentinel — all bits set to 1.
/// HDF5 uses this to indicate an unallocated or missing address.
pub const UNDEF_ADDR: u64 = u64::MAX;

/// A binary reader aware of the HDF5 file's `size_of_offsets` and `size_of_lengths`.
///
/// HDF5 is always little-endian, but address and length fields have variable
/// widths determined by the superblock.
#[derive(Debug, Clone)]
pub struct HDF5Reader {
    cursor: Cursor<Bytes>,
    size_of_offsets: u8,
    size_of_lengths: u8,
}

impl HDF5Reader {
    /// Create a reader for initial parsing (before superblock is read).
    /// Uses 8-byte offsets/lengths as a safe default.
    pub fn new(data: Bytes) -> Self {
        Self {
            cursor: Cursor::new(data),
            size_of_offsets: 8,
            size_of_lengths: 8,
        }
    }

    /// Create a reader with known superblock parameters.
    pub fn with_sizes(data: Bytes, size_of_offsets: u8, size_of_lengths: u8) -> Self {
        Self {
            cursor: Cursor::new(data),
            size_of_offsets,
            size_of_lengths,
        }
    }

    /// Current position in the buffer.
    pub fn position(&self) -> u64 {
        self.cursor.position()
    }

    /// Set the current position.
    pub fn set_position(&mut self, pos: u64) {
        self.cursor.set_position(pos);
    }

    /// Remaining bytes available.
    pub fn remaining(&self) -> usize {
        let pos = self.cursor.position() as usize;
        let len = self.cursor.get_ref().len();
        len.saturating_sub(pos)
    }

    /// The configured size of offset fields.
    pub fn size_of_offsets(&self) -> u8 {
        self.size_of_offsets
    }

    /// The configured size of length fields.
    pub fn size_of_lengths(&self) -> u8 {
        self.size_of_lengths
    }

    /// Read an offset field (variable width from superblock).
    pub fn read_offset(&mut self) -> Result<u64> {
        self.read_var_uint(self.size_of_offsets)
    }

    /// Read a length field (variable width from superblock).
    pub fn read_length(&mut self) -> Result<u64> {
        self.read_var_uint(self.size_of_lengths)
    }

    /// Check if an offset value represents the undefined address.
    pub fn is_undef_addr(addr: u64, size_of_offsets: u8) -> bool {
        let mask = if size_of_offsets >= 8 {
            u64::MAX
        } else {
            (1u64 << (size_of_offsets * 8)) - 1
        };
        addr == mask
    }

    /// Read a variable-width unsigned integer (1, 2, 4, or 8 bytes).
    fn read_var_uint(&mut self, width: u8) -> Result<u64> {
        match width {
            1 => Ok(self.read_u8()? as u64),
            2 => Ok(self.read_u16()? as u64),
            4 => Ok(self.read_u32()? as u64),
            8 => Ok(self.read_u64()?),
            _ => Err(HDF5Error::General(format!(
                "unsupported field width: {width}"
            ))),
        }
    }

    /// Read a `u8`.
    pub fn read_u8(&mut self) -> Result<u8> {
        Ok(self.cursor.read_u8()?)
    }

    /// Read a little-endian `u16`.
    pub fn read_u16(&mut self) -> Result<u16> {
        Ok(self.cursor.read_u16::<LittleEndian>()?)
    }

    /// Read a little-endian `u32`.
    pub fn read_u32(&mut self) -> Result<u32> {
        Ok(self.cursor.read_u32::<LittleEndian>()?)
    }

    /// Read a little-endian `u64`.
    pub fn read_u64(&mut self) -> Result<u64> {
        Ok(self.cursor.read_u64::<LittleEndian>()?)
    }

    /// Read an `i8`.
    pub fn read_i8(&mut self) -> Result<i8> {
        Ok(self.cursor.read_i8()?)
    }

    /// Read a little-endian `i16`.
    pub fn read_i16(&mut self) -> Result<i16> {
        Ok(self.cursor.read_i16::<LittleEndian>()?)
    }

    /// Read a little-endian `i32`.
    pub fn read_i32(&mut self) -> Result<i32> {
        Ok(self.cursor.read_i32::<LittleEndian>()?)
    }

    /// Read a little-endian `i64`.
    pub fn read_i64(&mut self) -> Result<i64> {
        Ok(self.cursor.read_i64::<LittleEndian>()?)
    }

    /// Read a little-endian `f32`.
    pub fn read_f32(&mut self) -> Result<f32> {
        Ok(self.cursor.read_f32::<LittleEndian>()?)
    }

    /// Read a little-endian `f64`.
    pub fn read_f64(&mut self) -> Result<f64> {
        Ok(self.cursor.read_f64::<LittleEndian>()?)
    }

    /// Read exactly `n` bytes.
    pub fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.cursor.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read a fixed-size magic/signature and compare.
    pub fn read_signature(&mut self, expected: &[u8]) -> Result<()> {
        let buf = self.read_bytes(expected.len())?;
        if buf != expected {
            Err(HDF5Error::General(format!(
                "invalid signature: expected {:?}, got {:?}",
                String::from_utf8_lossy(expected),
                String::from_utf8_lossy(&buf),
            )))
        } else {
            Ok(())
        }
    }

    /// Skip `n` bytes.
    pub fn skip(&mut self, n: u64) {
        self.cursor.set_position(self.cursor.position() + n);
    }

    /// Get a reference to the underlying bytes.
    pub fn get_ref(&self) -> &Bytes {
        self.cursor.get_ref()
    }

    /// Slice out bytes from the current position without advancing.
    pub fn slice_from_position(&self, len: usize) -> Result<Bytes> {
        let pos = self.cursor.position() as usize;
        let data = self.cursor.get_ref();
        if pos + len > data.len() {
            return Err(HDF5Error::UnexpectedEof {
                needed: len,
                available: data.len() - pos,
            });
        }
        Ok(data.slice(pos..pos + len))
    }
}
