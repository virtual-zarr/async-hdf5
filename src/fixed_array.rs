//! Fixed Array (FARRAY) chunk index reader.
//!
//! The Fixed Array is used for chunked datasets where all dimensions are fixed
//! (no unlimited dimensions). It provides O(1) chunk lookup.
//!
//! Structures:
//! - FAHD (Fixed Array Header) — metadata + pointer to data block
//! - FADB (Fixed Array Data Block) — array of chunk entries, optionally paged

use std::sync::Arc;

use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};
use crate::reader::AsyncFileReader;

/// Parsed Fixed Array header (FAHD).
#[derive(Debug)]
pub struct FixedArrayHeader {
    /// Client ID: 0 = non-filtered chunks, 1 = filtered chunks.
    pub client_id: u8,
    /// Size of each entry in bytes.
    pub entry_size: u8,
    /// Page bits — threshold for pagination (pages used if max_entries >= 2^page_bits).
    pub page_bits: u8,
    /// Maximum number of entries (total chunks).
    pub max_num_entries: u64,
    /// Address of the data block (FADB).
    pub data_block_address: u64,
}

/// A single chunk entry from a fixed or extensible array.
#[derive(Debug, Clone)]
pub(crate) struct FixedArrayChunkEntry {
    /// File byte offset of the chunk.
    pub address: u64,
    /// Chunk size in bytes (compressed/on-disk). For non-filtered, this is computed.
    pub chunk_size: u64,
    /// Filter mask (0 for non-filtered).
    pub filter_mask: u32,
}

impl FixedArrayHeader {
    /// Read and parse a Fixed Array header from the given address.
    pub async fn read(
        reader: &Arc<dyn AsyncFileReader>,
        address: u64,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        // FAHD is small: 4 (sig) + 1 (ver) + 1 (client) + 1 (entry_size) + 1 (page_bits)
        //   + L (max_entries) + O (data_block_addr) + 4 (checksum)
        let fetch_size = (12 + size_of_offsets + size_of_lengths) as u64;
        let data = reader.get_bytes(address..address + fetch_size).await?;
        let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);

        r.read_signature(b"FAHD")?;
        let version = r.read_u8()?;
        if version != 0 {
            return Err(HDF5Error::General(format!(
                "unsupported Fixed Array header version: {version}"
            )));
        }

        let client_id = r.read_u8()?;
        let entry_size = r.read_u8()?;
        let page_bits = r.read_u8()?;
        let max_num_entries = r.read_length()?;
        let data_block_address = r.read_offset()?;
        // Skip checksum

        Ok(Self {
            client_id,
            entry_size,
            page_bits,
            max_num_entries,
            data_block_address,
        })
    }
}

/// Read all chunk entries from a Fixed Array.
///
/// `layout_version` is the data layout message version (4 or 5) — this affects
/// how filtered chunk entries are encoded.
pub(crate) async fn read_fixed_array_entries(
    reader: &Arc<dyn AsyncFileReader>,
    header: &FixedArrayHeader,
    size_of_offsets: u8,
    size_of_lengths: u8,
    uncompressed_chunk_size: u64,
    layout_version: u8,
) -> Result<Vec<FixedArrayChunkEntry>> {
    if HDF5Reader::is_undef_addr(header.data_block_address, size_of_offsets) {
        return Ok(vec![]);
    }

    let _is_filtered = header.client_id == 1;
    let num_entries = header.max_num_entries as usize;

    // Determine if paging is used
    let use_paging = num_entries as u64 >= (1u64 << header.page_bits);

    // FADB header: 4 (sig) + 1 (ver) + 1 (client) + O (header_addr) + ...
    let fadb_header_size = 6 + size_of_offsets as u64;

    if use_paging {
        read_paged_entries(
            reader,
            header,
            size_of_offsets,
            size_of_lengths,
            uncompressed_chunk_size,
            layout_version,
            fadb_header_size,
        )
        .await
    } else {
        read_unpaged_entries(
            reader,
            header,
            size_of_offsets,
            size_of_lengths,
            uncompressed_chunk_size,
            layout_version,
            fadb_header_size,
        )
        .await
    }
}

/// Read entries from an unpaged data block (all entries inline).
async fn read_unpaged_entries(
    reader: &Arc<dyn AsyncFileReader>,
    header: &FixedArrayHeader,
    size_of_offsets: u8,
    size_of_lengths: u8,
    uncompressed_chunk_size: u64,
    layout_version: u8,
    fadb_header_size: u64,
) -> Result<Vec<FixedArrayChunkEntry>> {
    let num_entries = header.max_num_entries as usize;
    let entries_size = num_entries as u64 * header.entry_size as u64;
    let total_size = fadb_header_size + entries_size + 4; // +4 for checksum

    let data = reader
        .get_bytes(header.data_block_address..header.data_block_address + total_size)
        .await?;
    let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);

    // Parse FADB header
    r.read_signature(b"FADB")?;
    let version = r.read_u8()?;
    if version != 0 {
        return Err(HDF5Error::General(format!(
            "unsupported Fixed Array data block version: {version}"
        )));
    }
    let _client_id = r.read_u8()?;
    let _header_address = r.read_offset()?;

    // Read entries
    parse_entries(
        &mut r,
        num_entries,
        header.client_id == 1,
        size_of_offsets,
        uncompressed_chunk_size,
        layout_version,
    )
}

/// Read entries from a paged data block.
async fn read_paged_entries(
    reader: &Arc<dyn AsyncFileReader>,
    header: &FixedArrayHeader,
    size_of_offsets: u8,
    size_of_lengths: u8,
    uncompressed_chunk_size: u64,
    layout_version: u8,
    fadb_header_size: u64,
) -> Result<Vec<FixedArrayChunkEntry>> {
    let num_entries = header.max_num_entries as usize;
    let entries_per_page = 1usize << header.page_bits;
    let num_pages = num_entries.div_ceil(entries_per_page);
    let bitmap_size = num_pages.div_ceil(8);

    // Fetch the FADB header + bitmap
    let header_plus_bitmap = fadb_header_size + bitmap_size as u64 + 4; // +4 checksum
    let data = reader
        .get_bytes(header.data_block_address..header.data_block_address + header_plus_bitmap)
        .await?;
    let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);

    r.read_signature(b"FADB")?;
    let version = r.read_u8()?;
    if version != 0 {
        return Err(HDF5Error::General(format!(
            "unsupported Fixed Array data block version: {version}"
        )));
    }
    let _client_id = r.read_u8()?;
    let _header_address = r.read_offset()?;

    // Skip the page init bitmap (bitmap_size bytes). We always read all pages
    // unconditionally — see comment below.
    r.skip(bitmap_size as u64);

    // Pages follow immediately after the FADB (header + bitmap + checksum)
    let page_data_size = entries_per_page as u64 * header.entry_size as u64;
    let page_total_size = page_data_size + 4; // +4 checksum per page
    let pages_start = header.data_block_address + fadb_header_size + bitmap_size as u64 + 4;

    // HDF5 allocates all pages sequentially after the FADB regardless of the
    // page init bitmap. The bitmap is a write-tracking hint: a set bit means
    // the page has been written to. However, some HDF5 writers don't set the
    // bitmap correctly, so we always read every page. Truly unallocated chunks
    // within a page still have UNDEF_ADDR, which callers filter out.
    let mut all_entries = Vec::with_capacity(num_entries);
    let mut page_file_offset = pages_start;

    for page_idx in 0..num_pages {
        let entries_in_page = if page_idx == num_pages - 1 {
            num_entries - page_idx * entries_per_page
        } else {
            entries_per_page
        };

        // Read this page
        let page_data = reader
            .get_bytes(page_file_offset..page_file_offset + page_total_size)
            .await?;
        let mut pr = HDF5Reader::with_sizes(page_data, size_of_offsets, size_of_lengths);

        let page_entries = parse_entries(
            &mut pr,
            entries_in_page,
            header.client_id == 1,
            size_of_offsets,
            uncompressed_chunk_size,
            layout_version,
        )?;
        all_entries.extend(page_entries);

        page_file_offset += page_total_size;
    }

    Ok(all_entries)
}

/// Parse chunk entries from a reader.
pub(crate) fn parse_entries(
    r: &mut HDF5Reader,
    count: usize,
    is_filtered: bool,
    _size_of_offsets: u8,
    uncompressed_chunk_size: u64,
    layout_version: u8,
) -> Result<Vec<FixedArrayChunkEntry>> {
    let mut entries = Vec::with_capacity(count);

    for _ in 0..count {
        if is_filtered {
            let address = r.read_offset()?;

            let chunk_size = if layout_version >= 5 {
                // v5: chunk size uses size_of_offsets width
                r.read_offset()?
            } else {
                // v4: chunk size uses a variable encoding
                // "entry_size - size_of_offsets - 4" bytes for the size field
                let _size_field_len = (r.get_ref().len() as u64 - r.position()).min(8) as u8; // fallback
                                                                                              // Actually, the entry_size tells us exactly how many bytes per entry.
                                                                                              // entry_size = size_of_offsets + chunk_size_bytes + 4 (filter_mask)
                                                                                              // So chunk_size_bytes = entry_size - size_of_offsets - 4
                                                                                              // But we don't have entry_size here directly. We can compute it from
                                                                                              // the uncompressed chunk size: "one more than bytes needed to encode it"
                let nbytes = bytes_needed_for(uncompressed_chunk_size);
                read_n_byte_uint(r, nbytes)?
            };

            let filter_mask = r.read_u32()?;

            entries.push(FixedArrayChunkEntry {
                address,
                chunk_size,
                filter_mask,
            });
        } else {
            // Non-filtered: just an address
            let address = r.read_offset()?;

            entries.push(FixedArrayChunkEntry {
                address,
                chunk_size: uncompressed_chunk_size,
                filter_mask: 0,
            });
        }
    }

    Ok(entries)
}

/// Calculate how many bytes are needed to encode a value, plus one (HDF5 convention).
pub(crate) fn bytes_needed_for(value: u64) -> u8 {
    if value == 0 {
        return 1;
    }
    let bits = 64 - value.leading_zeros();
    let bytes = bits.div_ceil(8);
    // HDF5 convention: "one more than needed"
    (bytes as u8 + 1).min(8)
}

/// Read an N-byte unsigned integer (little-endian).
pub(crate) fn read_n_byte_uint(r: &mut HDF5Reader, n: u8) -> Result<u64> {
    let bytes = r.read_bytes(n as usize)?;
    let mut val = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        val |= (b as u64) << (i * 8);
    }
    Ok(val)
}
