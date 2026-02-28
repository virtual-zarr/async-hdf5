use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};

/// Well-known HDF5 filter IDs.
pub mod filter_ids {
    /// Deflate (zlib) compression.
    pub const DEFLATE: u16 = 1;
    /// Byte shuffle filter.
    pub const SHUFFLE: u16 = 2;
    /// Fletcher32 checksum.
    pub const FLETCHER32: u16 = 3;
    /// SZIP compression.
    pub const SZIP: u16 = 4;
    /// N-bit packing filter.
    pub const NBIT: u16 = 5;
    /// Scale-offset filter.
    pub const SCALEOFFSET: u16 = 6;
}

/// A single filter in the pipeline.
#[derive(Debug, Clone)]
pub struct Filter {
    /// Filter identification number.
    pub id: u16,
    /// Optional filter name (only for user-defined filters, id >= 256).
    pub name: Option<String>,
    /// Filter flags.
    pub flags: u16,
    /// Filter-specific parameters (e.g., deflate level, shuffle element size).
    pub client_data: Vec<u32>,
}

impl Filter {
    /// Human-readable name for well-known filters.
    pub fn display_name(&self) -> &str {
        match self.id {
            filter_ids::DEFLATE => "deflate",
            filter_ids::SHUFFLE => "shuffle",
            filter_ids::FLETCHER32 => "fletcher32",
            filter_ids::SZIP => "szip",
            filter_ids::NBIT => "nbit",
            filter_ids::SCALEOFFSET => "scaleoffset",
            _ => self.name.as_deref().unwrap_or("unknown"),
        }
    }
}

/// The filter pipeline describes the sequence of transformations applied to
/// chunk data before writing (and reversed on reading).
///
/// Message type 0x000B.
#[derive(Debug, Clone)]
pub struct FilterPipeline {
    /// The ordered list of filters in the pipeline.
    pub filters: Vec<Filter>,
}

impl FilterPipeline {
    /// An empty filter pipeline (no filters).
    pub fn empty() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    /// Parse from the raw message bytes.
    pub fn parse(data: &Bytes) -> Result<Self> {
        let mut r = HDF5Reader::new(data.clone());

        let version = r.read_u8()?;
        match version {
            1 => Self::parse_v1(&mut r),
            2 => Self::parse_v2(&mut r),
            _ => Err(HDF5Error::UnsupportedFilterPipelineVersion(version)),
        }
    }

    fn parse_v1(r: &mut HDF5Reader) -> Result<Self> {
        let num_filters = r.read_u8()? as usize;
        r.skip(6); // reserved

        let mut filters = Vec::with_capacity(num_filters);
        for _ in 0..num_filters {
            let id = r.read_u16()?;
            let name_length = r.read_u16()? as usize;
            let flags = r.read_u16()?;
            let num_client_data = r.read_u16()? as usize;

            let name = if name_length > 0 {
                let name_bytes = r.read_bytes(name_length)?;
                // Pad to 8-byte boundary
                let pad = (8 - (name_length % 8)) % 8;
                r.skip(pad as u64);
                Some(
                    String::from_utf8_lossy(&name_bytes)
                        .trim_end_matches('\0')
                        .to_string(),
                )
            } else {
                None
            };

            let mut client_data = Vec::with_capacity(num_client_data);
            for _ in 0..num_client_data {
                client_data.push(r.read_u32()?);
            }

            // Pad if odd number of client data values
            if num_client_data % 2 != 0 {
                r.skip(4);
            }

            filters.push(Filter {
                id,
                name,
                flags,
                client_data,
            });
        }

        Ok(Self { filters })
    }

    fn parse_v2(r: &mut HDF5Reader) -> Result<Self> {
        let num_filters = r.read_u8()? as usize;

        let mut filters = Vec::with_capacity(num_filters);
        for _ in 0..num_filters {
            let id = r.read_u16()?;

            // In v2, name is only present for non-standard filters (id >= 256)
            let name = if id >= 256 {
                let name_length = r.read_u16()? as usize;
                let name_bytes = r.read_bytes(name_length)?;
                Some(
                    String::from_utf8_lossy(&name_bytes)
                        .trim_end_matches('\0')
                        .to_string(),
                )
            } else {
                None
            };

            let flags = r.read_u16()?;
            let num_client_data = r.read_u16()? as usize;

            let mut client_data = Vec::with_capacity(num_client_data);
            for _ in 0..num_client_data {
                client_data.push(r.read_u32()?);
            }

            filters.push(Filter {
                id,
                name,
                flags,
                client_data,
            });
        }

        Ok(Self { filters })
    }
}
