//! Extensible Array (EA) chunk index reader.
//!
//! The Extensible Array is used for chunked datasets where the dataspace has a
//! single unlimited dimension. It provides efficient append-oriented storage
//! with O(1) lookup via a pre-balanced hierarchical structure.
//!
//! Structures:
//! - EAHD (Extensible Array Header) — configuration + pointer to index block
//! - EAIB (Extensible Array Index Block) — inline elements + data/super block addresses
//! - EASB (Extensible Array Super Block) — page bitmaps + data block addresses
//! - EADB (Extensible Array Data Block) — chunk entries, optionally paged

use std::sync::Arc;

use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};
use crate::fixed_array::{parse_entries, read_n_byte_uint, FixedArrayChunkEntry};
use crate::reader::AsyncFileReader;

/// Pre-computed information for a single super block level.
#[derive(Debug, Clone)]
pub struct SBlockInfo {
    /// Number of data blocks at this super block level.
    pub ndblks: usize,
    /// Number of elements in each data block at this level.
    pub dblk_nelmts: usize,
    /// Index of the first element covered by this super block level.
    pub start_idx: u64,
}

/// Parsed Extensible Array header (EAHD).
#[derive(Debug)]
pub struct ExtensibleArrayHeader {
    /// Client ID: 0 = non-filtered chunks, 1 = filtered chunks.
    pub client_id: u8,
    /// Size of each element in bytes.
    pub element_size: u8,
    /// Number of bits needed to store the max number of elements.
    pub max_nelmts_bits: u8,
    /// Number of elements stored directly in the index block.
    pub idx_blk_elmts: u8,
    /// Minimum number of elements per data block.
    pub data_blk_min_elmts: u8,
    /// Minimum number of data block pointers per super block.
    pub sup_blk_min_data_ptrs: u8,
    /// Log2 of max elements per data block page.
    pub max_dblk_page_nelmts_bits: u8,
    /// Address of the index block.
    pub index_block_address: u64,

    // Computed fields:
    /// Total number of super block levels.
    pub nsblks: usize,
    /// Info for each super block level.
    pub sblk_info: Vec<SBlockInfo>,
    /// Number of elements per data block page (0 means no paging).
    pub dblk_page_nelmts: usize,
    /// Byte width of array offset fields in EASB/EADB.
    pub arr_off_size: u8,
}

impl ExtensibleArrayHeader {
    /// Read and parse an Extensible Array header from the given address.
    pub async fn read(
        reader: &Arc<dyn AsyncFileReader>,
        address: u64,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        // EAHD size: 4 (sig) + 1 (ver) + 1 (client) + 1 (elem_size)
        //   + 1 (max_nelmts_bits) + 1 (idx_blk_elmts) + 1 (data_blk_min_elmts)
        //   + 1 (sup_blk_min_data_ptrs) + 1 (max_dblk_page_nelmts_bits)
        //   + 6*L (statistics) + O (index_block_address) + 4 (checksum)
        let fetch_size = 12 + 6 * size_of_lengths as u64 + size_of_offsets as u64 + 4;
        let data = reader.get_bytes(address..address + fetch_size).await?;
        let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);

        r.read_signature(b"EAHD")?;
        let version = r.read_u8()?;
        if version != 0 {
            return Err(HDF5Error::General(format!(
                "unsupported Extensible Array header version: {version}"
            )));
        }

        let client_id = r.read_u8()?;
        let element_size = r.read_u8()?;
        let max_nelmts_bits = r.read_u8()?;
        let idx_blk_elmts = r.read_u8()?;
        let data_blk_min_elmts = r.read_u8()?;
        let sup_blk_min_data_ptrs = r.read_u8()?;
        let max_dblk_page_nelmts_bits = r.read_u8()?;

        // Skip 6 statistics fields (each is size_of_lengths)
        r.skip(6 * size_of_lengths as u64);

        let index_block_address = r.read_offset()?;
        // Skip checksum

        // Compute super block info table (mirrors H5EA__hdr_init)
        let log2_min = log2_of_power_of_2(data_blk_min_elmts as u32);
        let nsblks = 1 + (max_nelmts_bits as usize - log2_min as usize);
        let dblk_page_nelmts = 1usize << max_dblk_page_nelmts_bits;
        let arr_off_size = (max_nelmts_bits + 7) / 8;

        let mut sblk_info = Vec::with_capacity(nsblks);
        let mut start_idx = 0u64;
        for s in 0..nsblks {
            let ndblks = 1usize << (s / 2);
            let dblk_nelmts = (1usize << ((s + 1) / 2)) * data_blk_min_elmts as usize;
            sblk_info.push(SBlockInfo {
                ndblks,
                dblk_nelmts,
                start_idx,
            });
            start_idx += (ndblks as u64) * (dblk_nelmts as u64);
        }

        Ok(Self {
            client_id,
            element_size,
            max_nelmts_bits,
            idx_blk_elmts,
            data_blk_min_elmts,
            sup_blk_min_data_ptrs,
            max_dblk_page_nelmts_bits,
            index_block_address,
            nsblks,
            sblk_info,
            dblk_page_nelmts,
            arr_off_size,
        })
    }
}

/// A single entry with its flat index in the EA.
pub(crate) struct IndexedEntry {
    pub flat_idx: u64,
    pub entry: FixedArrayChunkEntry,
}

/// Read all allocated chunk entries from an Extensible Array.
///
/// Returns `(flat_index, entry)` pairs for only allocated entries.
/// Unallocated super blocks and data blocks are skipped entirely.
pub(crate) async fn read_extensible_array_entries(
    reader: &Arc<dyn AsyncFileReader>,
    header: &ExtensibleArrayHeader,
    size_of_offsets: u8,
    size_of_lengths: u8,
    uncompressed_chunk_size: u64,
    layout_version: u8,
) -> Result<Vec<IndexedEntry>> {
    if HDF5Reader::is_undef_addr(header.index_block_address, size_of_offsets) {
        return Ok(vec![]);
    }

    let is_filtered = header.client_id == 1;

    // Compute index block layout (mirrors H5EA__iblock_alloc)
    let nsblks_in_iblock = if header.sup_blk_min_data_ptrs <= 1 {
        0usize
    } else {
        2 * log2_of_power_of_2(header.sup_blk_min_data_ptrs as u32) as usize
    };
    let ndblk_addrs = if header.sup_blk_min_data_ptrs <= 1 {
        0usize
    } else {
        2 * (header.sup_blk_min_data_ptrs as usize - 1)
    };
    let nsblk_addrs = header.nsblks.saturating_sub(nsblks_in_iblock);

    // Compute index block size to fetch
    let iblock_size = 4 // signature
        + 1 // version
        + 1 // client_id
        + size_of_offsets as usize // header address
        + header.idx_blk_elmts as usize * header.element_size as usize // inline elements
        + ndblk_addrs * size_of_offsets as usize // data block addresses
        + nsblk_addrs * size_of_offsets as usize // super block addresses
        + 4; // checksum

    let data = reader
        .get_bytes(
            header.index_block_address..header.index_block_address + iblock_size as u64,
        )
        .await?;
    let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);

    // Parse EAIB header
    r.read_signature(b"EAIB")?;
    let version = r.read_u8()?;
    if version != 0 {
        return Err(HDF5Error::General(format!(
            "unsupported Extensible Array index block version: {version}"
        )));
    }
    let _client_id = r.read_u8()?;
    let _header_address = r.read_offset()?;

    // Read inline elements (flat indices 0..idx_blk_elmts)
    let mut result = Vec::new();
    if header.idx_blk_elmts > 0 {
        let inline = parse_entries(
            &mut r,
            header.idx_blk_elmts as usize,
            is_filtered,
            size_of_offsets,
            uncompressed_chunk_size,
            layout_version,
        )?;
        for (i, entry) in inline.into_iter().enumerate() {
            if !HDF5Reader::is_undef_addr(entry.address, size_of_offsets) {
                result.push(IndexedEntry {
                    flat_idx: i as u64,
                    entry,
                });
            }
        }
    }

    // Read data block addresses stored directly in the index block
    let mut dblk_addrs = Vec::with_capacity(ndblk_addrs);
    for _ in 0..ndblk_addrs {
        dblk_addrs.push(r.read_offset()?);
    }

    // Read super block addresses
    let mut sblk_addrs = Vec::with_capacity(nsblk_addrs);
    for _ in 0..nsblk_addrs {
        sblk_addrs.push(r.read_offset()?);
    }

    // Process data blocks stored directly in the index block.
    // These correspond to the first `nsblks_in_iblock` super block levels.
    // Flat index starts after inline elements.
    let base_idx = header.idx_blk_elmts as u64;
    let mut dblk_idx = 0usize;
    for s in 0..nsblks_in_iblock.min(header.nsblks) {
        let info = &header.sblk_info[s];
        for d in 0..info.ndblks {
            if dblk_idx < dblk_addrs.len() {
                let addr = dblk_addrs[dblk_idx];
                dblk_idx += 1;
                let flat_start = base_idx + info.start_idx + (d as u64) * (info.dblk_nelmts as u64);
                collect_data_block_entries(
                    reader,
                    addr,
                    info.dblk_nelmts,
                    flat_start,
                    header,
                    size_of_offsets,
                    size_of_lengths,
                    uncompressed_chunk_size,
                    layout_version,
                    None,
                    &mut result,
                )
                .await?;
            }
        }
    }

    // Process super blocks (those stored by address in the index block)
    for (sblk_rel_idx, &sblk_addr) in sblk_addrs.iter().enumerate() {
        let sblk_idx = nsblks_in_iblock + sblk_rel_idx;
        if sblk_idx >= header.nsblks {
            break;
        }
        // Skip unallocated super blocks entirely
        if HDF5Reader::is_undef_addr(sblk_addr, size_of_offsets) {
            continue;
        }

        collect_super_block_entries(
            reader,
            sblk_addr,
            sblk_idx,
            base_idx,
            header,
            size_of_offsets,
            size_of_lengths,
            uncompressed_chunk_size,
            layout_version,
            &mut result,
        )
        .await?;
    }

    Ok(result)
}

/// Read an Extensible Array Super Block (EASB) and collect its data block entries.
async fn collect_super_block_entries(
    reader: &Arc<dyn AsyncFileReader>,
    address: u64,
    sblk_idx: usize,
    base_idx: u64,
    header: &ExtensibleArrayHeader,
    size_of_offsets: u8,
    size_of_lengths: u8,
    uncompressed_chunk_size: u64,
    layout_version: u8,
    result: &mut Vec<IndexedEntry>,
) -> Result<()> {
    let info = &header.sblk_info[sblk_idx];

    // Determine if data blocks at this level are paged
    let dblk_npages = if info.dblk_nelmts > header.dblk_page_nelmts {
        info.dblk_nelmts / header.dblk_page_nelmts
    } else {
        0
    };
    let dblk_page_init_size = if dblk_npages > 0 {
        (dblk_npages + 7) / 8
    } else {
        0
    };

    // Compute EASB size
    let sblock_size = 4 // signature
        + 1 // version
        + 1 // client_id
        + size_of_offsets as usize // header address
        + header.arr_off_size as usize // block offset
        + info.ndblks * dblk_page_init_size // page init bitmaps
        + info.ndblks * size_of_offsets as usize // data block addresses
        + 4; // checksum

    let data = reader
        .get_bytes(address..address + sblock_size as u64)
        .await?;
    let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);

    r.read_signature(b"EASB")?;
    let version = r.read_u8()?;
    if version != 0 {
        return Err(HDF5Error::General(format!(
            "unsupported Extensible Array super block version: {version}"
        )));
    }
    let _client_id = r.read_u8()?;
    let _header_address = r.read_offset()?;
    let _block_offset = read_n_byte_uint(&mut r, header.arr_off_size)?;

    // Read page init bitmaps (one per data block)
    let mut page_init_bitmaps: Vec<Vec<u8>> = Vec::with_capacity(info.ndblks);
    if dblk_page_init_size > 0 {
        for _ in 0..info.ndblks {
            let bitmap = r.read_bytes(dblk_page_init_size)?;
            page_init_bitmaps.push(bitmap);
        }
    }

    // Read data block addresses
    let mut dblk_addrs = Vec::with_capacity(info.ndblks);
    for _ in 0..info.ndblks {
        dblk_addrs.push(r.read_offset()?);
    }

    // Read each data block
    for (d, &dblk_addr) in dblk_addrs.iter().enumerate() {
        let page_bitmap = if !page_init_bitmaps.is_empty() {
            Some(&page_init_bitmaps[d])
        } else {
            None
        };

        let flat_start = base_idx + info.start_idx + (d as u64) * (info.dblk_nelmts as u64);
        collect_data_block_entries(
            reader,
            dblk_addr,
            info.dblk_nelmts,
            flat_start,
            header,
            size_of_offsets,
            size_of_lengths,
            uncompressed_chunk_size,
            layout_version,
            page_bitmap,
            result,
        )
        .await?;
    }

    Ok(())
}

/// Read an Extensible Array Data Block (EADB) and collect allocated entries.
async fn collect_data_block_entries(
    reader: &Arc<dyn AsyncFileReader>,
    address: u64,
    nelmts: usize,
    flat_start: u64,
    header: &ExtensibleArrayHeader,
    size_of_offsets: u8,
    size_of_lengths: u8,
    uncompressed_chunk_size: u64,
    layout_version: u8,
    page_bitmap: Option<&Vec<u8>>,
    result: &mut Vec<IndexedEntry>,
) -> Result<()> {
    if HDF5Reader::is_undef_addr(address, size_of_offsets) {
        return Ok(()); // Skip unallocated data blocks
    }

    let is_filtered = header.client_id == 1;
    let is_paged = nelmts > header.dblk_page_nelmts;

    // EADB prefix size
    let prefix_size = 4 // signature
        + 1 // version
        + 1 // client_id
        + size_of_offsets as usize // header address
        + header.arr_off_size as usize; // block offset

    if !is_paged {
        // Non-paged: elements are inline after prefix
        let total_size =
            prefix_size + nelmts * header.element_size as usize + 4; // +4 checksum
        let data = reader
            .get_bytes(address..address + total_size as u64)
            .await?;
        let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);

        r.read_signature(b"EADB")?;
        let version = r.read_u8()?;
        if version != 0 {
            return Err(HDF5Error::General(format!(
                "unsupported Extensible Array data block version: {version}"
            )));
        }
        let _client_id = r.read_u8()?;
        let _header_address = r.read_offset()?;
        let _block_offset = read_n_byte_uint(&mut r, header.arr_off_size)?;

        let entries = parse_entries(
            &mut r,
            nelmts,
            is_filtered,
            size_of_offsets,
            uncompressed_chunk_size,
            layout_version,
        )?;
        for (i, entry) in entries.into_iter().enumerate() {
            if !HDF5Reader::is_undef_addr(entry.address, size_of_offsets) {
                result.push(IndexedEntry {
                    flat_idx: flat_start + i as u64,
                    entry,
                });
            }
        }
    } else {
        // Paged: read prefix only (no inline elements), then pages follow
        let prefix_total = prefix_size + 4; // +4 checksum
        let data = reader
            .get_bytes(address..address + prefix_total as u64)
            .await?;
        let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);

        r.read_signature(b"EADB")?;
        let version = r.read_u8()?;
        if version != 0 {
            return Err(HDF5Error::General(format!(
                "unsupported Extensible Array data block version: {version}"
            )));
        }
        let _client_id = r.read_u8()?;
        let _header_address = r.read_offset()?;
        let _block_offset = read_n_byte_uint(&mut r, header.arr_off_size)?;

        // Pages follow after the data block (prefix + checksum)
        let npages = nelmts / header.dblk_page_nelmts;
        let page_size =
            header.dblk_page_nelmts * header.element_size as usize + 4; // +4 checksum per page
        let pages_start = address + prefix_total as u64;

        for page_idx in 0..npages {
            // Check page init bitmap if available
            let page_initialized = match page_bitmap {
                Some(bitmap) => {
                    let byte_idx = page_idx / 8;
                    let bit_idx = page_idx % 8;
                    byte_idx < bitmap.len() && (bitmap[byte_idx] >> bit_idx) & 1 != 0
                }
                None => true,
            };

            if !page_initialized {
                continue; // Skip uninitialized pages
            }

            let page_addr = pages_start + (page_idx as u64) * (page_size as u64);
            let page_data = reader
                .get_bytes(page_addr..page_addr + page_size as u64)
                .await?;
            let mut pr = HDF5Reader::with_sizes(page_data, size_of_offsets, size_of_lengths);

            let page_entries = parse_entries(
                &mut pr,
                header.dblk_page_nelmts,
                is_filtered,
                size_of_offsets,
                uncompressed_chunk_size,
                layout_version,
            )?;
            let page_flat_start = flat_start + (page_idx * header.dblk_page_nelmts) as u64;
            for (i, entry) in page_entries.into_iter().enumerate() {
                if !HDF5Reader::is_undef_addr(entry.address, size_of_offsets) {
                    result.push(IndexedEntry {
                        flat_idx: page_flat_start + i as u64,
                        entry,
                    });
                }
            }
        }
    }

    Ok(())
}

/// Compute log2 of a power of 2. Panics if `n` is not a power of 2.
fn log2_of_power_of_2(n: u32) -> u32 {
    debug_assert!(n > 0 && n.is_power_of_two(), "{n} is not a power of 2");
    n.trailing_zeros()
}
