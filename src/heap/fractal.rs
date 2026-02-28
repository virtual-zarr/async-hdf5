use std::sync::Arc;

use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};
use crate::reader::AsyncFileReader;

/// Parsed fractal heap header.
///
/// Fractal heaps store variable-length objects (link messages, attributes) for
/// v2 groups and dense attribute storage. Objects are referenced by heap IDs
/// that encode the offset and length within the heap.
///
/// Three object types:
/// - **Managed**: Stored in direct blocks within the doubling table.
/// - **Huge**: Stored directly in the file, referenced by B-tree.
/// - **Tiny**: Embedded directly in the heap ID (very small objects).
#[derive(Debug)]
pub struct FractalHeap {
    reader: Arc<dyn AsyncFileReader>,
    size_of_offsets: u8,
    size_of_lengths: u8,

    /// Size of heap IDs for this heap.
    id_length: u16,
    /// Maximum size of managed objects.
    #[allow(dead_code)]
    max_managed_object_size: u32,
    /// Starting block size in the doubling table.
    starting_block_size: u64,
    /// Maximum direct block size.
    max_direct_block_size: u64,
    /// Maximum heap size as log2(bytes) — determines offset field width in IDs.
    max_heap_size: u16,
    /// Table width (number of direct block slots per row).
    table_width: u16,
    /// Address of the root block (direct or indirect).
    root_block_address: u64,
    /// Current number of rows in root indirect block. 0 = root is a direct block.
    current_root_rows: u16,
    /// Whether direct blocks have checksums.
    #[allow(dead_code)]
    checksum_direct_blocks: bool,
    /// Number of managed objects.
    #[allow(dead_code)]
    num_managed_objects: u64,
    /// Filtered root direct block size (if I/O filters present).
    #[allow(dead_code)]
    filtered_root_size: Option<u64>,
    /// I/O filter mask (if filters present).
    #[allow(dead_code)]
    filter_mask: Option<u32>,
    /// I/O filters encoded length.
    io_filters_length: u16,
}

impl FractalHeap {
    /// Read and parse a fractal heap header from the file.
    pub async fn read(
        reader: &Arc<dyn AsyncFileReader>,
        address: u64,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        // Fractal heap header is typically 88-120 bytes.
        let fetch_size = 256u64;
        let data = reader.get_bytes(address..address + fetch_size).await?;
        let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);

        let sig = r.read_bytes(4)?;
        if &sig != b"FRHP" {
            return Err(HDF5Error::InvalidHeapSignature {
                expected: "FRHP".into(),
                got: String::from_utf8_lossy(&sig).into(),
            });
        }

        let version = r.read_u8()?;
        if version != 0 {
            return Err(HDF5Error::UnsupportedHeapVersion(version));
        }

        let id_length = r.read_u16()?;
        let io_filters_length = r.read_u16()?;
        let flags = r.read_u8()?;

        let checksum_direct_blocks = flags & 0x02 != 0;

        let max_managed_object_size = r.read_u32()?;
        let _next_huge_id = r.read_length()?;
        let _huge_btree_address = r.read_offset()?;
        let _free_space = r.read_length()?;
        let _free_space_manager_address = r.read_offset()?;
        let _managed_space = r.read_length()?;
        let _allocated_managed_space = r.read_length()?;
        let _direct_block_alloc_iterator_offset = r.read_length()?;
        let num_managed_objects = r.read_length()?;
        let _huge_objects_size = r.read_length()?;
        let _num_huge_objects = r.read_length()?;
        let _tiny_objects_size = r.read_length()?;
        let _num_tiny_objects = r.read_length()?;

        let table_width = r.read_u16()?;
        let starting_block_size = r.read_length()?;
        let max_direct_block_size = r.read_length()?;
        let max_heap_size = r.read_u16()?;
        let _starting_root_rows = r.read_u16()?;
        let root_block_address = r.read_offset()?;
        let current_root_rows = r.read_u16()?;

        let (filtered_root_size, filter_mask) = if io_filters_length > 0 {
            let size = r.read_length()?;
            let mask = r.read_u32()?;
            // Skip the I/O filter information bytes
            r.skip(io_filters_length as u64);
            (Some(size), Some(mask))
        } else {
            (None, None)
        };

        Ok(Self {
            reader: Arc::clone(reader),
            size_of_offsets,
            size_of_lengths,
            id_length,
            max_managed_object_size,
            starting_block_size,
            max_direct_block_size,
            max_heap_size,
            table_width,
            root_block_address,
            current_root_rows,
            checksum_direct_blocks,
            num_managed_objects,
            filtered_root_size,
            filter_mask,
            io_filters_length,
        })
    }

    /// Decode a fractal heap ID and return the referenced object bytes.
    pub async fn get_object(&self, heap_id: &[u8]) -> Result<Bytes> {
        if heap_id.is_empty() {
            return Err(HDF5Error::General("empty fractal heap ID".into()));
        }

        let first_byte = heap_id[0];
        let id_type = (first_byte >> 4) & 0x03;

        match id_type {
            0 => self.get_managed_object(heap_id).await,
            1 => Err(HDF5Error::General(
                "huge fractal heap objects not yet supported".into(),
            )),
            2 => self.get_tiny_object(heap_id),
            _ => Err(HDF5Error::General(format!(
                "unknown fractal heap ID type: {id_type}"
            ))),
        }
    }

    /// Get a managed object by decoding offset and length from the heap ID.
    async fn get_managed_object(&self, heap_id: &[u8]) -> Result<Bytes> {
        // Managed ID layout: [type_byte][offset...][length...]
        // Offset field size = ceil(max_heap_size / 8) bytes
        // Length field size = remaining bytes in the ID
        let offset_size = ((self.max_heap_size as usize) + 7) / 8;
        let length_size = self.id_length as usize - 1 - offset_size;

        if heap_id.len() < 1 + offset_size + length_size {
            return Err(HDF5Error::General("fractal heap ID too short".into()));
        }

        let offset = read_var_uint(&heap_id[1..1 + offset_size]);
        let length = read_var_uint(&heap_id[1 + offset_size..1 + offset_size + length_size]);

        if length == 0 {
            return Ok(Bytes::new());
        }

        // Navigate the doubling table to find the direct block containing this offset.
        self.read_from_offset(offset, length).await
    }

    /// Read `length` bytes from the given linear offset within the heap.
    async fn read_from_offset(&self, offset: u64, length: u64) -> Result<Bytes> {
        if self.current_root_rows == 0 {
            // Root block is a direct block — the offset is directly within it.
            // The heap's address space includes the block header, so the offset
            // from the heap ID already accounts for the header bytes.
            let file_offset = self.root_block_address + offset;
            return self
                .reader
                .get_bytes(file_offset..file_offset + length)
                .await
                .map_err(Into::into);
        }

        // Root block is an indirect block — traverse the doubling table.
        self.read_from_indirect_block(self.root_block_address, self.current_root_rows, offset, length)
            .await
    }

    /// Read from an indirect block by navigating down to the correct direct block.
    async fn read_from_indirect_block(
        &self,
        iblock_address: u64,
        nrows: u16,
        offset: u64,
        length: u64,
    ) -> Result<Bytes> {
        // Calculate direct block row info
        let max_dblock_rows = self.max_direct_block_rows();
        let ndirect_rows = nrows.min(max_dblock_rows);
        let nindirect_rows = nrows.saturating_sub(max_dblock_rows);
        let ndirect_children = ndirect_rows as usize * self.table_width as usize;
        let nindirect_children = nindirect_rows as usize * self.table_width as usize;

        // Calculate the block offset field size
        let block_offset_size = ((self.max_heap_size as usize) + 7) / 8;

        // Indirect block header: FHIB(4) + version(1) + heap_header_addr(O) + block_offset(variable)
        let iblock_header_size =
            4 + 1 + self.size_of_offsets as usize + block_offset_size;

        // Each direct child entry: address(O) [+ filtered_size(L) + filter_mask(4) if filtered]
        let direct_entry_size = if self.io_filters_length > 0 {
            self.size_of_offsets as usize + self.size_of_lengths as usize + 4
        } else {
            self.size_of_offsets as usize
        };

        // Each indirect child entry: address(O)
        let indirect_entry_size = self.size_of_offsets as usize;

        let total_size = iblock_header_size
            + ndirect_children * direct_entry_size
            + nindirect_children * indirect_entry_size
            + 4; // checksum

        let data = self
            .reader
            .get_bytes(iblock_address..iblock_address + total_size as u64)
            .await?;
        let mut r = HDF5Reader::with_sizes(data, self.size_of_offsets, self.size_of_lengths);

        // Skip header
        r.read_signature(b"FHIB")?;
        let _version = r.read_u8()?;
        let _heap_header_addr = r.read_offset()?;
        r.skip(block_offset_size as u64); // block offset

        // Build list of direct block addresses and their sizes
        let mut cumulative_offset = 0u64;
        for row in 0..ndirect_rows {
            let block_size = self.row_block_size(row as usize);
            for _col in 0..self.table_width {
                let block_addr = r.read_offset()?;
                if self.io_filters_length > 0 {
                    let _filtered_size = r.read_length()?;
                    let _filter_mask = r.read_u32()?;
                }

                if offset >= cumulative_offset && offset < cumulative_offset + block_size {
                    // Found the right direct block
                    if HDF5Reader::is_undef_addr(block_addr, self.size_of_offsets) {
                        return Err(HDF5Error::UndefinedAddress);
                    }
                    // The heap's address space includes block headers, so the
                    // local offset already accounts for them.
                    let local_offset = offset - cumulative_offset;
                    let file_offset = block_addr + local_offset;
                    return self
                        .reader
                        .get_bytes(file_offset..file_offset + length)
                        .await
                        .map_err(Into::into);
                }
                cumulative_offset += block_size;
            }
        }

        // If not found in direct blocks, check indirect blocks
        for row in 0..nindirect_rows {
            let iblock_row = max_dblock_rows as usize + row as usize;
            let block_size = self.row_block_size(iblock_row);
            for _col in 0..self.table_width {
                let child_iblock_addr = r.read_offset()?;

                if offset >= cumulative_offset && offset < cumulative_offset + block_size {
                    if HDF5Reader::is_undef_addr(child_iblock_addr, self.size_of_offsets) {
                        return Err(HDF5Error::UndefinedAddress);
                    }
                    let local_offset = offset - cumulative_offset;
                    let child_rows = self.rows_for_block_size(block_size);
                    return Box::pin(self.read_from_indirect_block(
                        child_iblock_addr,
                        child_rows,
                        local_offset,
                        length,
                    ))
                    .await;
                }
                cumulative_offset += block_size;
            }
        }

        Err(HDF5Error::General(format!(
            "fractal heap offset {offset} out of range (max {cumulative_offset})"
        )))
    }

    /// Maximum number of rows in any indirect block that contain direct block children.
    fn max_direct_block_rows(&self) -> u16 {
        if self.starting_block_size == 0 {
            return 0;
        }
        let log2_start = (self.starting_block_size as f64).log2() as u16;
        let log2_max_direct = (self.max_direct_block_size as f64).log2() as u16;
        (log2_max_direct - log2_start) + 2
    }

    /// Block size for a given row index in the doubling table.
    fn row_block_size(&self, row: usize) -> u64 {
        if row < 2 {
            self.starting_block_size
        } else {
            self.starting_block_size * (1u64 << (row - 1))
        }
    }

    /// Number of indirect block rows needed for a given block size.
    fn rows_for_block_size(&self, block_size: u64) -> u16 {
        if block_size <= self.max_direct_block_size {
            return 0;
        }
        let log2_start = (self.starting_block_size as f64).log2() as u16;
        let log2_block = (block_size as f64).log2() as u16;
        (log2_block - log2_start) + 1
    }

    /// Get a tiny object — data is embedded directly in the heap ID.
    fn get_tiny_object(&self, heap_id: &[u8]) -> Result<Bytes> {
        let first_byte = heap_id[0];
        let data_length;
        let data_start;

        if self.id_length <= 18 {
            // Normal tiny: lower 4 bits = length - 1
            data_length = ((first_byte & 0x0F) + 1) as usize;
            data_start = 1;
        } else {
            // Extended tiny: lower 4 bits = high nibble of 12-bit length-1
            let high = (first_byte & 0x0F) as usize;
            if heap_id.len() < 2 {
                return Err(HDF5Error::General("tiny heap ID too short".into()));
            }
            let low = heap_id[1] as usize;
            data_length = ((high << 8) | low) + 1;
            data_start = 2;
        };

        if heap_id.len() < data_start + data_length {
            return Err(HDF5Error::General("tiny heap ID data truncated".into()));
        }

        Ok(Bytes::copy_from_slice(
            &heap_id[data_start..data_start + data_length],
        ))
    }
}

/// Read a variable-width little-endian unsigned integer from a byte slice.
fn read_var_uint(bytes: &[u8]) -> u64 {
    let mut value = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        value |= (b as u64) << (i * 8);
    }
    value
}
