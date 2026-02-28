use std::sync::Arc;

use crate::btree;
use crate::chunk_index::{ChunkIndex, ChunkLocation};
use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};
use crate::extensible_array;
use crate::fixed_array;
use crate::group::attributes_from_header;
use crate::messages::attribute::Attribute;
use crate::messages::data_layout::{ChunkIndexParams, ChunkIndexType, StorageLayout};
use crate::messages::dataspace::DataspaceMessage;
use crate::messages::datatype::DataType;
use crate::messages::fill_value::FillValueMessage;
use crate::messages::filter_pipeline::FilterPipeline;
use crate::object_header::{msg_types, ObjectHeader};
use crate::reader::AsyncFileReader;
use crate::superblock::Superblock;

/// An HDF5 dataset — a typed, shaped, chunked (or contiguous) array of data.
///
/// Provides synchronous access to parsed metadata (shape, dtype, chunk shape,
/// filters, fill value) and async access to the chunk index (byte offsets of
/// all chunks in the file).
#[derive(Debug)]
pub struct HDF5Dataset {
    name: String,
    header: ObjectHeader,
    reader: Arc<dyn AsyncFileReader>,
    superblock: Arc<Superblock>,

    // Parsed metadata (cached on construction)
    shape: Vec<u64>,
    dtype: DataType,
    layout: StorageLayout,
    filters: FilterPipeline,
    fill_value: Option<Vec<u8>>,
}

impl HDF5Dataset {
    /// Create a new dataset by parsing metadata messages from its object header.
    pub fn new(
        name: String,
        header: ObjectHeader,
        reader: Arc<dyn AsyncFileReader>,
        superblock: Arc<Superblock>,
    ) -> Result<Self> {
        // Parse dataspace
        let dataspace = header
            .find_message(msg_types::DATASPACE)
            .ok_or_else(|| HDF5Error::General("dataset missing dataspace message".into()))?;
        let dataspace = DataspaceMessage::parse(&dataspace.data, superblock.size_of_lengths)?;

        // Parse datatype
        let dtype_msg = header
            .find_message(msg_types::DATATYPE)
            .ok_or_else(|| HDF5Error::General("dataset missing datatype message".into()))?;
        let dtype = DataType::parse(&dtype_msg.data)?;

        // Parse data layout
        let layout_msg = header
            .find_message(msg_types::DATA_LAYOUT)
            .ok_or_else(|| HDF5Error::General("dataset missing data layout message".into()))?;
        let layout = StorageLayout::parse(
            &layout_msg.data,
            superblock.size_of_offsets,
            superblock.size_of_lengths,
        )?;

        // Parse filter pipeline (optional — absent means no filters)
        let filters = if let Some(filter_msg) = header.find_message(msg_types::FILTER_PIPELINE) {
            FilterPipeline::parse(&filter_msg.data)?
        } else {
            FilterPipeline::empty()
        };

        // Parse fill value (optional)
        let fill_value = if let Some(fv_msg) = header.find_message(msg_types::FILL_VALUE) {
            FillValueMessage::parse(&fv_msg.data)?.value
        } else if let Some(fv_msg) = header.find_message(msg_types::FILL_VALUE_OLD) {
            // Old fill value message: size(4) + data
            let mut r = HDF5Reader::new(fv_msg.data.clone());
            let size = r.read_u32()? as usize;
            if size > 0 {
                Some(r.read_bytes(size)?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            name,
            header,
            reader,
            superblock,
            shape: dataspace.dimensions,
            dtype,
            layout,
            filters,
            fill_value,
        })
    }

    /// The dataset's name (not the full path).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Dataset shape (current dimensions).
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Data type.
    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    /// Element size in bytes.
    pub fn element_size(&self) -> u32 {
        self.dtype.size()
    }

    /// Chunk shape (None for contiguous or compact storage).
    pub fn chunk_shape(&self) -> Option<&[u64]> {
        match &self.layout {
            StorageLayout::Chunked { chunk_shape, .. } => Some(chunk_shape),
            _ => None,
        }
    }

    /// Filter pipeline.
    pub fn filters(&self) -> &FilterPipeline {
        &self.filters
    }

    /// Fill value bytes (interpretation depends on dtype).
    pub fn fill_value(&self) -> Option<&[u8]> {
        self.fill_value.as_deref()
    }

    /// Storage layout.
    pub fn layout(&self) -> &StorageLayout {
        &self.layout
    }

    /// Access the object header.
    pub fn header(&self) -> &ObjectHeader {
        &self.header
    }

    /// Get all attributes attached to this dataset, resolving vlen data.
    pub async fn attributes(&self) -> Vec<Attribute> {
        attributes_from_header(
            &self.header,
            &self.reader,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await
    }

    /// Get a single attribute by name.
    pub async fn attribute(&self, name: &str) -> Option<Attribute> {
        self.attributes().await.into_iter().find(|a| a.name == name)
    }

    /// Extract the chunk index — the key async operation.
    ///
    /// For chunked datasets, traverses the B-tree to enumerate all chunks.
    /// For contiguous datasets, returns a single-entry index.
    /// For compact datasets, returns an empty index (data is inline in the header).
    pub async fn chunk_index(&self) -> Result<ChunkIndex> {
        match &self.layout {
            StorageLayout::Compact { .. } => {
                // Compact: data is in the object header. No file-level chunks.
                Ok(ChunkIndex::new(vec![], vec![], self.shape.clone()))
            }

            StorageLayout::Contiguous { address, size } => {
                if HDF5Reader::is_undef_addr(*address, self.superblock.size_of_offsets) {
                    // Unallocated (never written) — empty index
                    return Ok(ChunkIndex::new(vec![], vec![], self.shape.clone()));
                }
                Ok(ChunkIndex::contiguous(*address, *size, self.shape.clone()))
            }

            StorageLayout::Chunked {
                chunk_shape,
                index_address,
                indexing_type,
                index_params,
                flags,
            } => {
                if HDF5Reader::is_undef_addr(*index_address, self.superblock.size_of_offsets) {
                    // Unallocated — no chunks written yet
                    return Ok(ChunkIndex::new(
                        vec![],
                        chunk_shape.clone(),
                        self.shape.clone(),
                    ));
                }

                match indexing_type {
                    ChunkIndexType::BTreeV1 => {
                        self.chunk_index_btree_v1(*index_address, chunk_shape)
                            .await
                    }
                    ChunkIndexType::BTreeV2 => {
                        self.chunk_index_btree_v2(*index_address, chunk_shape)
                            .await
                    }
                    ChunkIndexType::SingleChunk => {
                        self.chunk_index_single_chunk(
                            *index_address,
                            chunk_shape,
                            index_params,
                        )
                    }
                    ChunkIndexType::FixedArray => {
                        self.chunk_index_fixed_array(
                            *index_address,
                            chunk_shape,
                            *flags,
                        )
                        .await
                    }
                    ChunkIndexType::ExtensibleArray => {
                        self.chunk_index_extensible_array(
                            *index_address,
                            chunk_shape,
                            *flags,
                        )
                        .await
                    }
                    other => Err(HDF5Error::General(format!(
                        "chunk indexing type {other:?} not yet supported"
                    ))),
                }
            }
        }
    }

    /// Extract chunk index from a B-tree v1 (legacy chunked datasets).
    async fn chunk_index_btree_v1(
        &self,
        btree_address: u64,
        chunk_shape: &[u64],
    ) -> Result<ChunkIndex> {
        let ndims = self.shape.len();

        let entries = btree::v1::read_chunk_btree_v1(
            &self.reader,
            btree_address,
            ndims,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        // Convert B-tree v1 entries to ChunkLocation.
        // v1 chunk keys store offsets in *elements* (not scaled). We need to
        // convert to chunk indices by dividing by chunk dimensions.
        let locations: Vec<ChunkLocation> = entries
            .into_iter()
            .map(|e| {
                let indices: Vec<u64> = e
                    .offsets
                    .iter()
                    .zip(chunk_shape.iter())
                    .map(|(&offset, &cs)| offset / cs)
                    .collect();
                ChunkLocation {
                    indices,
                    byte_offset: e.address,
                    byte_length: e.size as u64,
                    filter_mask: e.filter_mask,
                }
            })
            .collect();

        Ok(ChunkIndex::new(
            locations,
            chunk_shape.to_vec(),
            self.shape.clone(),
        ))
    }

    /// Extract chunk index from a B-tree v2 (modern chunked datasets).
    async fn chunk_index_btree_v2(
        &self,
        btree_address: u64,
        chunk_shape: &[u64],
    ) -> Result<ChunkIndex> {
        let ndims = self.shape.len();

        let header = btree::v2::BTreeV2Header::read(
            &self.reader,
            btree_address,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        let raw_records = btree::v2::collect_all_records(
            &self.reader,
            &header,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        match header.record_type {
            11 => {
                // Filtered chunks (most common for compressed datasets like NISAR)
                let chunk_records = btree::v2::parse_chunk_records_filtered(
                    &raw_records,
                    ndims,
                    self.superblock.size_of_offsets,
                    self.superblock.size_of_lengths,
                )?;

                let locations: Vec<ChunkLocation> = chunk_records
                    .into_iter()
                    .filter(|c| {
                        !HDF5Reader::is_undef_addr(c.address, self.superblock.size_of_offsets)
                    })
                    .map(|c| ChunkLocation {
                        indices: c.scaled_offsets,
                        byte_offset: c.address,
                        byte_length: c.chunk_size,
                        filter_mask: c.filter_mask,
                    })
                    .collect();

                Ok(ChunkIndex::new(
                    locations,
                    chunk_shape.to_vec(),
                    self.shape.clone(),
                ))
            }
            10 => {
                // Non-filtered chunks
                let chunk_records = btree::v2::parse_chunk_records_non_filtered(
                    &raw_records,
                    ndims,
                    self.superblock.size_of_offsets,
                )?;

                // For non-filtered, every chunk has the same uncompressed size
                let uncompressed_chunk_size: u64 =
                    chunk_shape.iter().product::<u64>() * self.dtype.size() as u64;

                let locations: Vec<ChunkLocation> = chunk_records
                    .into_iter()
                    .filter(|c| {
                        !HDF5Reader::is_undef_addr(c.address, self.superblock.size_of_offsets)
                    })
                    .map(|c| ChunkLocation {
                        indices: c.scaled_offsets,
                        byte_offset: c.address,
                        byte_length: uncompressed_chunk_size,
                        filter_mask: 0,
                    })
                    .collect();

                Ok(ChunkIndex::new(
                    locations,
                    chunk_shape.to_vec(),
                    self.shape.clone(),
                ))
            }
            other => Err(HDF5Error::General(format!(
                "B-tree v2 record type {other} not supported for chunk indexing (expected 10 or 11)"
            ))),
        }
    }

    /// Extract chunk index from a Fixed Array (type 3).
    async fn chunk_index_fixed_array(
        &self,
        fahd_address: u64,
        chunk_shape: &[u64],
        layout_flags: u8,
    ) -> Result<ChunkIndex> {
        let ndims = self.shape.len();

        // Read the FAHD header
        let fahd = fixed_array::FixedArrayHeader::read(
            &self.reader,
            fahd_address,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        // Compute uncompressed chunk size for non-filtered entries
        let uncompressed_chunk_size =
            chunk_shape.iter().product::<u64>() * self.dtype.size() as u64;

        // Determine layout version from the data layout message
        // We need to know if it's v4 or v5 for filtered entry encoding.
        // The header version was parsed earlier; use flags bit 1 for filtered detection.
        let layout_version = if layout_flags & 0x02 != 0 { 4u8 } else { 4u8 };

        // Read all entries from the FADB
        let entries = fixed_array::read_fixed_array_entries(
            &self.reader,
            &fahd,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
            uncompressed_chunk_size,
            layout_version,
        )
        .await?;

        // Convert to ChunkLocations. The entries are stored in row-major order
        // of the chunk grid.
        let grid_shape: Vec<u64> = self
            .shape
            .iter()
            .zip(chunk_shape.iter())
            .map(|(&ds, &cs)| (ds + cs - 1) / cs)
            .collect();

        let mut locations = Vec::new();
        for (flat_idx, entry) in entries.iter().enumerate() {
            // Skip undefined addresses (unallocated chunks)
            if HDF5Reader::is_undef_addr(entry.address, self.superblock.size_of_offsets) {
                continue;
            }

            // Convert flat index to multi-dimensional indices (row-major)
            let mut indices = vec![0u64; ndims];
            let mut remaining = flat_idx as u64;
            for d in (0..ndims).rev() {
                indices[d] = remaining % grid_shape[d];
                remaining /= grid_shape[d];
            }

            locations.push(ChunkLocation {
                indices,
                byte_offset: entry.address,
                byte_length: entry.chunk_size,
                filter_mask: entry.filter_mask,
            });
        }

        Ok(ChunkIndex::new(
            locations,
            chunk_shape.to_vec(),
            self.shape.clone(),
        ))
    }

    /// Extract chunk index from an Extensible Array (type 4).
    async fn chunk_index_extensible_array(
        &self,
        eahd_address: u64,
        chunk_shape: &[u64],
        layout_flags: u8,
    ) -> Result<ChunkIndex> {
        let ndims = self.shape.len();

        // Read the EAHD header
        let eahd = extensible_array::ExtensibleArrayHeader::read(
            &self.reader,
            eahd_address,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        // Compute uncompressed chunk size for non-filtered entries
        let uncompressed_chunk_size =
            chunk_shape.iter().product::<u64>() * self.dtype.size() as u64;

        let layout_version = if layout_flags & 0x02 != 0 { 4u8 } else { 4u8 };

        // Read all entries from the EA structure
        let entries = extensible_array::read_extensible_array_entries(
            &self.reader,
            &eahd,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
            uncompressed_chunk_size,
            layout_version,
        )
        .await?;

        // Convert to ChunkLocations using flat indices from EA
        let grid_shape: Vec<u64> = self
            .shape
            .iter()
            .zip(chunk_shape.iter())
            .map(|(&ds, &cs)| (ds + cs - 1) / cs)
            .collect();

        let mut locations = Vec::with_capacity(entries.len());
        for indexed in &entries {
            let mut indices = vec![0u64; ndims];
            let mut remaining = indexed.flat_idx;
            for d in (0..ndims).rev() {
                indices[d] = remaining % grid_shape[d];
                remaining /= grid_shape[d];
            }

            locations.push(ChunkLocation {
                indices,
                byte_offset: indexed.entry.address,
                byte_length: indexed.entry.chunk_size,
                filter_mask: indexed.entry.filter_mask,
            });
        }

        Ok(ChunkIndex::new(
            locations,
            chunk_shape.to_vec(),
            self.shape.clone(),
        ))
    }

    /// Handle single-chunk datasets (chunk indexing type 1).
    fn chunk_index_single_chunk(
        &self,
        address: u64,
        chunk_shape: &[u64],
        index_params: &ChunkIndexParams,
    ) -> Result<ChunkIndex> {
        let ndims = self.shape.len();

        let (byte_length, filter_mask) = match index_params {
            ChunkIndexParams::SingleChunk {
                filtered_size,
                filter_mask,
            } => (*filtered_size, *filter_mask),
            _ => {
                // Unfiltered single chunk — compute size from shape and element size
                let size = chunk_shape.iter().product::<u64>() * self.dtype.size() as u64;
                (size, 0u32)
            }
        };

        let location = ChunkLocation {
            indices: vec![0; ndims],
            byte_offset: address,
            byte_length,
            filter_mask,
        };

        Ok(ChunkIndex::new(
            vec![location],
            chunk_shape.to_vec(),
            self.shape.clone(),
        ))
    }
}
