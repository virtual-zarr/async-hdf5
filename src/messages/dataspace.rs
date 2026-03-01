use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::Result;

/// Dataspace message — describes the dimensionality of a dataset.
///
/// Message type 0x0001.
#[derive(Debug, Clone)]
pub struct DataspaceMessage {
    /// Number of dimensions (0 = scalar).
    pub rank: u8,
    /// Dataspace type: 0 = scalar, 1 = simple, 2 = null.
    /// Only present for v2 messages; v1 defaults to 1 (simple) for rank > 0,
    /// 0 (scalar) for rank == 0.
    pub dataspace_type: u8,
    /// Current size along each dimension.
    pub dimensions: Vec<u64>,
    /// Maximum size along each dimension (None = same as current).
    /// A value of u64::MAX means unlimited.
    pub max_dimensions: Option<Vec<u64>>,
}

impl DataspaceMessage {
    /// Parse from the raw message bytes.
    pub fn parse(data: &Bytes, size_of_lengths: u8) -> Result<Self> {
        let mut r = HDF5Reader::with_sizes(data.clone(), 8, size_of_lengths);

        let version = r.read_u8()?;
        let rank = r.read_u8()?;
        let flags = r.read_u8()?;

        // v1: reserved (1 byte) + reserved (4 bytes) = 5 bytes (no type field)
        // v2: type field (1 byte): 0=scalar, 1=simple, 2=null
        let dataspace_type = match version {
            1 => {
                r.skip(5); // 1 reserved + 4 reserved
                if rank == 0 { 0 } else { 1 }
            }
            2 => {
                r.read_u8()?
            }
            _ => {
                // Best-effort: try to continue
                
                r.read_u8()?
            }
        };

        let mut dimensions = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            dimensions.push(r.read_length()?);
        }

        let has_max = flags & 0x01 != 0;
        let max_dimensions = if has_max {
            let mut max = Vec::with_capacity(rank as usize);
            for _ in 0..rank {
                max.push(r.read_length()?);
            }
            Some(max)
        } else {
            None
        };

        Ok(Self {
            rank,
            dataspace_type,
            dimensions,
            max_dimensions,
        })
    }
}
