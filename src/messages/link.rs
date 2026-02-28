use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};

/// Link type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkType {
    /// Hard link — points directly to an object header.
    Hard,
    /// Soft link — symbolic reference by path string.
    Soft,
    /// External link — reference to an object in another file.
    External,
}

/// A link message — describes a single named link within a group (v2 groups).
///
/// Message type 0x0006.
#[derive(Debug, Clone)]
pub struct LinkMessage {
    /// Link name.
    pub name: String,
    /// Link type.
    pub link_type: LinkType,
    /// For hard links: address of the target object header.
    pub target_address: Option<u64>,
    /// For soft links: the link value string.
    pub soft_link_value: Option<String>,
    /// Creation order (if tracked).
    pub creation_order: Option<u64>,
}

impl LinkMessage {
    /// Parse from the raw message bytes.
    pub fn parse(data: &Bytes, size_of_offsets: u8, size_of_lengths: u8) -> Result<Self> {
        let mut r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);

        let _version = r.read_u8()?;
        let flags = r.read_u8()?;

        // Flags bits:
        // 0-1: Size of link name length field (0=1, 1=2, 2=4, 3=8 bytes)
        // 2: Creation Order field present
        // 3: Link Type field present
        // 4: Link Name Character Set field present

        let link_type = if flags & 0x08 != 0 {
            match r.read_u8()? {
                0 => LinkType::Hard,
                1 => LinkType::Soft,
                64 => LinkType::External,
                t => return Err(HDF5Error::UnsupportedLinkType(t)),
            }
        } else {
            LinkType::Hard // default
        };

        let creation_order = if flags & 0x04 != 0 {
            Some(r.read_u64()?)
        } else {
            None
        };

        let _charset = if flags & 0x10 != 0 {
            r.read_u8()? // 0=ASCII, 1=UTF-8
        } else {
            0
        };

        let name_size_field_width = 1u8 << (flags & 0x03);
        let name_length = match name_size_field_width {
            1 => r.read_u8()? as usize,
            2 => r.read_u16()? as usize,
            4 => r.read_u32()? as usize,
            8 => r.read_u64()? as usize,
            _ => unreachable!(),
        };

        let name_bytes = r.read_bytes(name_length)?;
        let name = String::from_utf8_lossy(&name_bytes).to_string();

        let (target_address, soft_link_value) = match link_type {
            LinkType::Hard => {
                let addr = r.read_offset()?;
                (Some(addr), None)
            }
            LinkType::Soft => {
                let value_length = r.read_u16()? as usize;
                let value_bytes = r.read_bytes(value_length)?;
                let value = String::from_utf8_lossy(&value_bytes).to_string();
                (None, Some(value))
            }
            LinkType::External => {
                let value_length = r.read_u16()? as usize;
                let value_bytes = r.read_bytes(value_length)?;
                let value = String::from_utf8_lossy(&value_bytes).to_string();
                (None, Some(value))
            }
        };

        Ok(Self {
            name,
            link_type,
            target_address,
            soft_link_value,
            creation_order,
        })
    }
}
