use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};

/// Chunk indexing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkIndexType {
    /// Type 1: single chunk — address points directly to it.
    SingleChunk,
    /// Type 2: implicit index — fixed grid, computed addresses.
    Implicit,
    /// Type 3: fixed array — one unlimited dimension.
    FixedArray,
    /// Type 4: extensible array — multiple unlimited dimensions.
    ExtensibleArray,
    /// Type 5: B-tree v2 — general case.
    BTreeV2,
    /// Legacy: layout message v3 uses B-tree v1.
    BTreeV1,
}

/// Data layout message — describes how dataset storage is organized.
///
/// Message type 0x0008.
#[derive(Debug, Clone)]
pub enum StorageLayout {
    /// Data stored inline in the object header. For very small datasets.
    Compact {
        data: Bytes,
    },

    /// Data stored as a single contiguous block in the file.
    Contiguous {
        /// Byte offset in the file. UNDEF_ADDR = unallocated.
        address: u64,
        /// Total size in bytes.
        size: u64,
    },

    /// Data stored in chunks, with an index structure for lookup.
    Chunked {
        /// Chunk dimensions in array elements.
        chunk_shape: Vec<u64>,
        /// Address of the chunk index (B-tree or other structure).
        index_address: u64,
        /// How chunks are indexed.
        indexing_type: ChunkIndexType,
        /// Layout flags.
        flags: u8,
        /// Additional indexing parameters (type-specific).
        index_params: ChunkIndexParams,
    },
}

/// Type-specific chunk index parameters.
#[derive(Debug, Clone)]
pub enum ChunkIndexParams {
    /// No extra params needed (B-tree v1, implicit).
    None,
    /// Single chunk: filtered chunk size and filter mask.
    SingleChunk {
        filtered_size: u64,
        filter_mask: u32,
    },
    /// Fixed array: page bits.
    FixedArray {
        page_bits: u8,
    },
    /// Extensible array: max bits, index elements, min pointers, min elements, page bits.
    ExtensibleArray {
        max_bits: u8,
        index_elements: u8,
        min_pointers: u8,
        min_elements: u8,
        page_bits: u8,
    },
}

impl StorageLayout {
    /// Parse from the raw data layout message bytes.
    pub fn parse(data: &Bytes, size_of_offsets: u8, size_of_lengths: u8) -> Result<Self> {
        let mut r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);

        let version = r.read_u8()?;

        match version {
            1 | 2 => Self::parse_v1_v2(&mut r, version),
            3 => Self::parse_v3(&mut r, size_of_offsets, size_of_lengths),
            4 | 5 => Self::parse_v4_v5(&mut r, version, size_of_offsets, size_of_lengths),
            _ => Err(HDF5Error::UnsupportedDataLayoutVersion(version)),
        }
    }

    /// Parse layout message version 1 or 2.
    fn parse_v1_v2(r: &mut HDF5Reader, version: u8) -> Result<Self> {
        let ndims = r.read_u8()?;
        let layout_class = r.read_u8()?;
        r.skip(5); // reserved

        // v1 has data address here for all classes
        if version == 1 {
            let _data_address = r.read_u32()?;
        }

        let address = r.read_offset()?;

        match layout_class {
            0 => {
                // Compact
                let data_size = r.read_u32()? as usize;
                let data = r.slice_from_position(data_size)?;
                r.skip(data_size as u64);
                Ok(StorageLayout::Compact { data })
            }
            1 => {
                // Contiguous
                let mut total_size = 1u64;
                for _ in 0..ndims {
                    total_size *= r.read_u32()? as u64;
                }
                let element_size = r.read_u32()? as u64;
                Ok(StorageLayout::Contiguous {
                    address,
                    size: total_size * element_size,
                })
            }
            2 => {
                // Chunked
                let mut chunk_shape = Vec::with_capacity(ndims as usize);
                // v1/v2: ndims dimensions, each 4 bytes, last one is element size
                for _ in 0..ndims.saturating_sub(1) {
                    chunk_shape.push(r.read_u32()? as u64);
                }
                let _element_size = r.read_u32()?;

                Ok(StorageLayout::Chunked {
                    chunk_shape,
                    index_address: address,
                    indexing_type: ChunkIndexType::BTreeV1,
                    flags: 0,
                    index_params: ChunkIndexParams::None,
                })
            }
            _ => Err(HDF5Error::General(format!(
                "unknown layout class: {layout_class}"
            ))),
        }
    }

    /// Parse layout message version 3.
    fn parse_v3(
        r: &mut HDF5Reader,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        let layout_class = r.read_u8()?;

        match layout_class {
            0 => {
                // Compact
                let data_size = r.read_u16()? as usize;
                let data = r.slice_from_position(data_size)?;
                r.skip(data_size as u64);
                Ok(StorageLayout::Compact { data })
            }
            1 => {
                // Contiguous
                let address = r.read_offset()?;
                let size = r.read_length()?;
                Ok(StorageLayout::Contiguous { address, size })
            }
            2 => {
                // Chunked (v3 = B-tree v1)
                let ndims = r.read_u8()?;
                let address = r.read_offset()?;

                // ndims dimension sizes (4 bytes each), last is element size
                let mut chunk_shape = Vec::with_capacity(ndims as usize);
                for _ in 0..ndims.saturating_sub(1) {
                    chunk_shape.push(r.read_u32()? as u64);
                }
                let _element_size = r.read_u32()?;

                Ok(StorageLayout::Chunked {
                    chunk_shape,
                    index_address: address,
                    indexing_type: ChunkIndexType::BTreeV1,
                    flags: 0,
                    index_params: ChunkIndexParams::None,
                })
            }
            _ => Err(HDF5Error::General(format!(
                "unknown layout class: {layout_class}"
            ))),
        }
    }

    /// Parse layout message version 4 or 5 (modern, with chunk indexing types).
    fn parse_v4_v5(
        r: &mut HDF5Reader,
        version: u8,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        let layout_class = r.read_u8()?;

        match layout_class {
            0 => {
                // Compact
                let data_size = r.read_u16()? as usize;
                let data = r.slice_from_position(data_size)?;
                r.skip(data_size as u64);
                Ok(StorageLayout::Compact { data })
            }
            1 => {
                // Contiguous
                let address = r.read_offset()?;
                let size = r.read_length()?;
                Ok(StorageLayout::Contiguous { address, size })
            }
            2 => {
                // Chunked with indexing type
                let flags = r.read_u8()?;
                let ndims = r.read_u8()?;
                let dim_size_enc_len = r.read_u8()?;

                // ndims includes an extra dimension for the element size,
                // just like v3. We read all ndims values but only keep the
                // first ndims-1 as the actual chunk shape.
                let mut chunk_shape = Vec::with_capacity(ndims as usize);
                for _ in 0..ndims {
                    let dim = match dim_size_enc_len {
                        1 => r.read_u8()? as u64,
                        2 => r.read_u16()? as u64,
                        4 => r.read_u32()? as u64,
                        8 => r.read_u64()?,
                        _ => {
                            return Err(HDF5Error::General(format!(
                                "unsupported dimension size encoding length: {dim_size_enc_len}"
                            )));
                        }
                    };
                    chunk_shape.push(dim);
                }
                // Last dimension is element size, not a chunk dimension
                let _element_size = chunk_shape.pop();

                let chunk_indexing_type = r.read_u8()?;
                let (indexing_type, index_params) = match chunk_indexing_type {
                    1 => {
                        // Single chunk
                        let params = if flags & 0x02 != 0 {
                            // Filtered single chunk
                            let filtered_size = r.read_length()?;
                            let filter_mask = r.read_u32()?;
                            ChunkIndexParams::SingleChunk {
                                filtered_size,
                                filter_mask,
                            }
                        } else {
                            ChunkIndexParams::None
                        };
                        (ChunkIndexType::SingleChunk, params)
                    }
                    2 => {
                        // Implicit
                        (ChunkIndexType::Implicit, ChunkIndexParams::None)
                    }
                    3 => {
                        // Fixed array
                        let page_bits = r.read_u8()?;
                        (
                            ChunkIndexType::FixedArray,
                            ChunkIndexParams::FixedArray { page_bits },
                        )
                    }
                    4 => {
                        // Extensible array
                        let max_bits = r.read_u8()?;
                        let index_elements = r.read_u8()?;
                        let min_pointers = r.read_u8()?;
                        let min_elements = r.read_u8()?;
                        let page_bits = r.read_u8()?;
                        (
                            ChunkIndexType::ExtensibleArray,
                            ChunkIndexParams::ExtensibleArray {
                                max_bits,
                                index_elements,
                                min_pointers,
                                min_elements,
                                page_bits,
                            },
                        )
                    }
                    5 => {
                        // B-tree v2
                        (ChunkIndexType::BTreeV2, ChunkIndexParams::None)
                    }
                    _ => {
                        return Err(HDF5Error::UnsupportedChunkIndexingType(chunk_indexing_type));
                    }
                };

                let index_address = r.read_offset()?;

                Ok(StorageLayout::Chunked {
                    chunk_shape,
                    index_address,
                    indexing_type,
                    flags,
                    index_params,
                })
            }
            _ => Err(HDF5Error::General(format!(
                "unknown layout class: {layout_class}"
            ))),
        }
    }

    /// Returns true if the layout is chunked.
    pub fn is_chunked(&self) -> bool {
        matches!(self, StorageLayout::Chunked { .. })
    }

    /// Returns true if the layout is contiguous.
    pub fn is_contiguous(&self) -> bool {
        matches!(self, StorageLayout::Contiguous { .. })
    }
}
