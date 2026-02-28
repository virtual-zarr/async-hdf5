use std::sync::Arc;

use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::Result;
use crate::group::HDF5Group;
use crate::object_header::ObjectHeader;
use crate::reader::{AsyncFileReader, BlockCache};
use crate::superblock::Superblock;

/// An opened HDF5 file.
///
/// This is the main entry point for reading HDF5 metadata. It holds a
/// reference to the async reader and the parsed superblock.
///
/// # Example
///
/// ```ignore
/// use async_hdf5::HDF5File;
/// use async_hdf5::reader::ObjectReader;
///
/// let reader = ObjectReader::new(store, path);
/// let file = HDF5File::open(reader).await?;
/// let root = file.root_group_header().await?;
/// ```
#[derive(Debug)]
pub struct HDF5File {
    reader: Arc<dyn AsyncFileReader>,
    superblock: Superblock,
}

impl HDF5File {
    /// Open an HDF5 file by parsing its superblock.
    ///
    /// Wraps the given reader in a `BlockCache` (default 8 MiB blocks) to
    /// coalesce the many small metadata reads into a few large requests.
    pub async fn open(reader: impl AsyncFileReader) -> Result<Self> {
        Self::open_with_block_size(reader, 8 * 1024 * 1024).await
    }

    /// Open with a configurable block cache size.
    pub async fn open_with_block_size(
        reader: impl AsyncFileReader,
        block_size: u64,
    ) -> Result<Self> {
        let cached = BlockCache::new(reader).with_block_size(block_size);

        let initial_bytes = cached.get_bytes(0..block_size.min(64 * 1024)).await?;
        let (superblock, _offset) = Superblock::parse(&initial_bytes)?;

        Ok(Self {
            reader: Arc::new(cached),
            superblock,
        })
    }

    /// Open with an already-configured reader (e.g., a pre-built `BlockCache`).
    pub async fn open_raw(reader: Arc<dyn AsyncFileReader>) -> Result<Self> {
        let initial_bytes = reader.get_bytes(0..64 * 1024).await?;
        let (superblock, _offset) = Superblock::parse(&initial_bytes)?;

        Ok(Self {
            reader,
            superblock,
        })
    }

    /// Access the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Access the underlying async reader.
    pub fn reader(&self) -> &Arc<dyn AsyncFileReader> {
        &self.reader
    }

    /// Read and parse the object header at the given file address.
    pub async fn read_object_header(&self, address: u64) -> Result<ObjectHeader> {
        read_object_header(
            &self.reader,
            address,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await
    }

    /// Read the root group's object header.
    pub async fn root_group_header(&self) -> Result<ObjectHeader> {
        self.read_object_header(self.superblock.root_group_address)
            .await
    }

    /// Get the root group as an `HDF5Group` for navigation.
    pub async fn root_group(&self) -> Result<HDF5Group> {
        let header = self.root_group_header().await?;
        Ok(HDF5Group::new(
            "/".to_string(),
            header,
            Arc::clone(&self.reader),
            Arc::new(self.superblock.clone()),
        ))
    }
}

/// Read and parse an object header, following any continuation messages.
pub(crate) async fn read_object_header(
    reader: &Arc<dyn AsyncFileReader>,
    address: u64,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<ObjectHeader> {
    // Initial fetch — 4 KB is usually enough for one object header.
    let initial_size = 4096u64;
    let data = reader.get_bytes(address..address + initial_size).await?;

    // Check if the first chunk is larger than our initial fetch and re-fetch if needed.
    let needed = peek_object_header_size(&data)?;
    let data = if needed > initial_size {
        reader.get_bytes(address..address + needed).await?
    } else {
        data
    };

    let mut header = ObjectHeader::parse(&data, size_of_offsets, size_of_lengths)?;

    // Follow continuation messages iteratively — continuation chunks can
    // themselves contain further continuation messages (nested chains).
    let mut pending = header.continuation_addresses(size_of_offsets, size_of_lengths)?;
    while let Some((cont_addr, cont_len)) = pending.pop() {
        let cont_data = reader.get_bytes(cont_addr..cont_addr + cont_len).await?;
        let new_messages = parse_continuation_chunk(
            &cont_data,
            size_of_offsets,
            size_of_lengths,
            header.version,
            header.track_creation_order,
        )?;

        // Check new messages for further continuations before adding them
        for msg in &new_messages {
            if msg.msg_type == crate::object_header::msg_types::HEADER_CONTINUATION {
                let mut r = HDF5Reader::with_sizes(
                    msg.data.clone(),
                    size_of_offsets,
                    size_of_lengths,
                );
                let address = r.read_offset()?;
                let length = r.read_length()?;
                pending.push((address, length));
            }
        }

        header.messages.extend(new_messages);
    }

    Ok(header)
}

/// Peek at an object header's prefix to determine the total size of the first chunk.
///
/// For v2 headers: reads the flags and chunk0_size to compute prefix + chunk0_size.
/// For v1 headers: reads the header_size field to compute 16 + header_size.
fn peek_object_header_size(data: &Bytes) -> Result<u64> {
    if data.len() < 6 {
        return Err(crate::error::HDF5Error::General(
            "object header data too short".into(),
        ));
    }

    if data.len() >= 4 && data[0..4] == [b'O', b'H', b'D', b'R'] {
        // v2 header
        let flags = data[5];
        let mut offset = 6usize;

        // Optional timestamps (flags bit 5)
        if flags & 0x20 != 0 {
            offset += 16;
        }

        // Optional attribute phase change values (flags bit 4)
        if flags & 0x10 != 0 {
            offset += 4;
        }

        // Chunk size field width
        let chunk_size_width = 1usize << (flags & 0x03);
        if data.len() < offset + chunk_size_width {
            return Err(crate::error::HDF5Error::General(
                "object header too short for chunk size field".into(),
            ));
        }

        let chunk0_size = match chunk_size_width {
            1 => data[offset] as u64,
            2 => u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as u64,
            4 => u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as u64,
            8 => u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()),
            _ => unreachable!(),
        };

        // Total = header prefix + chunk0_size (which includes messages + checksum)
        Ok((offset + chunk_size_width) as u64 + chunk0_size)
    } else {
        // v1 header: version(1) + reserved(1) + num_messages(2) + ref_count(4) +
        //            header_size(4) + reserved(4) = 16 bytes prefix
        if data.len() < 12 {
            return Err(crate::error::HDF5Error::General(
                "v1 object header too short".into(),
            ));
        }
        let header_size = u32::from_le_bytes(data[8..12].try_into().unwrap()) as u64;
        Ok(16 + header_size)
    }
}

/// Parse messages from a continuation chunk (OCHK in v2, raw messages in v1).
fn parse_continuation_chunk(
    data: &Bytes,
    size_of_offsets: u8,
    size_of_lengths: u8,
    header_version: u8,
    track_creation_order: bool,
) -> Result<Vec<crate::object_header::HeaderMessage>> {
    use crate::object_header::msg_types;

    let mut r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);
    let mut messages = Vec::new();

    if header_version == 2 {
        // v2 continuation: starts with OCHK signature
        r.read_signature(&[b'O', b'C', b'H', b'K'])?;

        let end = data.len() as u64 - 4; // minus checksum
        while r.position() < end {
            let msg_type = r.read_u8()? as u16;
            let msg_size = r.read_u16()? as usize;
            let flags = r.read_u8()?;

            // Skip creation order field if tracked (same format as primary chunk)
            if track_creation_order {
                let _creation_order = r.read_u16()?;
            }

            // NIL message (type 0) signals start of gap/padding to end of chunk.
            if msg_type == msg_types::NIL {
                break;
            }

            let msg_data = if msg_size > 0 {
                let d = r.slice_from_position(msg_size)?;
                r.skip(msg_size as u64);
                d
            } else {
                Bytes::new()
            };

            messages.push(crate::object_header::HeaderMessage {
                msg_type,
                data: msg_data,
                flags,
            });
        }
    } else {
        // v1 continuation: raw messages with 8-byte alignment
        let end = data.len() as u64;
        while r.position() + 8 <= end {
            let msg_type = r.read_u16()?;
            let msg_size = r.read_u16()? as usize;
            let flags = r.read_u8()?;
            r.skip(3); // reserved

            if msg_size == 0 && msg_type == msg_types::NIL {
                let pad = (8 - (r.position() % 8)) % 8;
                r.skip(pad);
                continue;
            }

            let msg_data = if msg_size > 0 {
                let d = r.slice_from_position(msg_size)?;
                r.skip(msg_size as u64);
                d
            } else {
                Bytes::new()
            };

            let pad = (8 - (r.position() % 8)) % 8;
            r.skip(pad);

            messages.push(crate::object_header::HeaderMessage {
                msg_type,
                data: msg_data,
                flags,
            });
        }
    }

    Ok(messages)
}
