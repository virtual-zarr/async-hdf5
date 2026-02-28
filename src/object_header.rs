use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};

/// Object Header v2 signature.
const OHDR_SIGNATURE: [u8; 4] = [b'O', b'H', b'D', b'R'];

/// Object Header Continuation signature (v2).
#[allow(dead_code)]
const OCHK_SIGNATURE: [u8; 4] = [b'O', b'C', b'H', b'K'];

/// A raw header message (type + data bytes, not yet parsed into a specific message).
#[derive(Debug, Clone)]
pub struct HeaderMessage {
    /// Message type ID.
    pub msg_type: u16,
    /// Raw message data (interpretation depends on type).
    pub data: Bytes,
    /// Message flags.
    pub flags: u8,
}

/// Known header message types.
pub mod msg_types {
    /// NIL message (padding).
    pub const NIL: u16 = 0x0000;
    /// Dataspace message — describes array dimensionality.
    pub const DATASPACE: u16 = 0x0001;
    /// Link info message — fractal heap + B-tree v2 for group links.
    pub const LINK_INFO: u16 = 0x0002;
    /// Datatype message — describes element data type.
    pub const DATATYPE: u16 = 0x0003;
    /// Fill value (old) message.
    pub const FILL_VALUE_OLD: u16 = 0x0004;
    /// Fill value message.
    pub const FILL_VALUE: u16 = 0x0005;
    /// Link message — describes a single link in a group.
    pub const LINK: u16 = 0x0006;
    /// External data files message.
    pub const EXTERNAL_DATA_FILES: u16 = 0x0007;
    /// Data layout message — compact/contiguous/chunked storage.
    pub const DATA_LAYOUT: u16 = 0x0008;
    /// Bogus message (testing).
    pub const BOGUS: u16 = 0x0009;
    /// Group info message.
    pub const GROUP_INFO: u16 = 0x000A;
    /// Filter pipeline message.
    pub const FILTER_PIPELINE: u16 = 0x000B;
    /// Attribute message.
    pub const ATTRIBUTE: u16 = 0x000C;
    /// Object comment message.
    pub const OBJECT_COMMENT: u16 = 0x000D;
    /// Object modification time (old) message.
    pub const MODIFICATION_TIME_OLD: u16 = 0x000E;
    /// Shared message table message.
    pub const SHARED_MESSAGE_TABLE: u16 = 0x000F;
    /// Object header continuation message.
    pub const HEADER_CONTINUATION: u16 = 0x0010;
    /// Symbol table message (v1 groups).
    pub const SYMBOL_TABLE: u16 = 0x0011;
    /// Object modification time message.
    pub const MODIFICATION_TIME: u16 = 0x0012;
    /// B-tree 'K' values message.
    pub const BTREE_K_VALUES: u16 = 0x0013;
    /// Driver info message.
    pub const DRIVER_INFO: u16 = 0x0014;
    /// Attribute info message.
    pub const ATTRIBUTE_INFO: u16 = 0x0015;
    /// Object reference count message.
    pub const REFERENCE_COUNT: u16 = 0x0016;
}

/// Parsed object header containing its version and all raw messages.
#[derive(Debug, Clone)]
pub struct ObjectHeader {
    /// Object header version (1 or 2).
    pub version: u8,
    /// Whether creation order is tracked for messages (v2 only, flags bit 2).
    pub track_creation_order: bool,
    /// All header messages, in order.
    pub messages: Vec<HeaderMessage>,
}

impl ObjectHeader {
    /// Parse an object header from a byte buffer at the given position.
    ///
    /// For v2 headers, the data must start with the `OHDR` signature.
    /// For v1 headers, there is no signature — the version byte comes first.
    ///
    /// `continuation_fetcher` is called when a continuation message is found
    /// and we need to fetch more data from a different file offset. For the
    /// initial parse (from a single prefetched buffer), pass `None`.
    pub fn parse(data: &Bytes, size_of_offsets: u8, size_of_lengths: u8) -> Result<Self> {
        let _r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);

        // Peek at first bytes to determine version
        if data.len() >= 4 && data[0..4] == OHDR_SIGNATURE {
            Self::parse_v2(data, size_of_offsets, size_of_lengths)
        } else {
            Self::parse_v1(data, size_of_offsets, size_of_lengths)
        }
    }

    /// Parse a version 1 object header.
    ///
    /// Layout:
    ///   - Version (1 byte) — value 1
    ///   - Reserved (1 byte)
    ///   - Number of Header Messages (2 bytes)
    ///   - Object Reference Count (4 bytes)
    ///   - Object Header Size (4 bytes)
    ///   - Reserved/Padding (4 bytes, align to 8)
    ///   - Messages...
    fn parse_v1(data: &Bytes, size_of_offsets: u8, size_of_lengths: u8) -> Result<Self> {
        let mut r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);

        let version = r.read_u8()?;
        if version != 1 {
            return Err(HDF5Error::UnsupportedObjectHeaderVersion(version));
        }

        let _reserved = r.read_u8()?;
        let num_messages = r.read_u16()?;
        let _ref_count = r.read_u32()?;
        let header_size = r.read_u32()? as u64;
        // Align to 8 bytes — skip padding (4 bytes of reserved in some interpretations)
        let _reserved2 = r.read_u32()?;

        let msg_start = r.position();
        let msg_end = msg_start + header_size;

        let mut messages = Vec::with_capacity(num_messages as usize);

        while r.position() < msg_end && messages.len() < num_messages as usize {
            // v1 message header: type(2) + size(2) + flags(1) + reserved(3)
            let msg_type = r.read_u16()?;
            let msg_size = r.read_u16()? as usize;
            let flags = r.read_u8()?;
            r.skip(3); // reserved

            if msg_size == 0 && msg_type == msg_types::NIL {
                continue;
            }

            let msg_data = r.slice_from_position(msg_size)?;
            r.skip(msg_size as u64);

            // Align to 8-byte boundary
            let pad = (8 - (r.position() % 8)) % 8;
            r.skip(pad);

            messages.push(HeaderMessage {
                msg_type,
                data: msg_data,
                flags,
            });
        }

        Ok(Self {
            version: 1,
            track_creation_order: false,
            messages,
        })
    }

    /// Parse a version 2 object header.
    ///
    /// Layout:
    ///   - Signature "OHDR" (4 bytes)
    ///   - Version (1 byte) — value 2
    ///   - Flags (1 byte)
    ///   - [optional timestamps, phase change values based on flags]
    ///   - Size of Chunk #0 (variable: 1, 2, 4, or 8 bytes)
    ///   - Messages... (packed, no alignment padding)
    ///   - Checksum (4 bytes) — gap byte 0 marks end before checksum
    fn parse_v2(data: &Bytes, size_of_offsets: u8, size_of_lengths: u8) -> Result<Self> {
        let mut r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);

        r.read_signature(&OHDR_SIGNATURE)?;
        let version = r.read_u8()?;
        if version != 2 {
            return Err(HDF5Error::UnsupportedObjectHeaderVersion(version));
        }

        let flags = r.read_u8()?;

        // Optional timestamps (if flags bit 5 set)
        if flags & 0x20 != 0 {
            let _access_time = r.read_u32()?;
            let _modification_time = r.read_u32()?;
            let _change_time = r.read_u32()?;
            let _birth_time = r.read_u32()?;
        }

        // Optional attribute phase change values (if flags bit 4 set)
        if flags & 0x10 != 0 {
            let _max_compact = r.read_u16()?;
            let _min_dense = r.read_u16()?;
        }

        // Size of chunk #0
        let chunk_size_width = 1u8 << (flags & 0x03);
        let chunk0_size = match chunk_size_width {
            1 => r.read_u8()? as u64,
            2 => r.read_u16()? as u64,
            4 => r.read_u32()? as u64,
            8 => r.read_u64()?,
            _ => unreachable!(),
        };

        let track_creation_order = flags & 0x04 != 0;
        let msg_start = r.position();
        let msg_end = msg_start + chunk0_size - 4; // -4 for the trailing checksum

        let mut messages = Vec::new();

        while r.position() < msg_end {
            // v2 message: type(1) + size(2) + flags(1) + [creation_order(2)]
            let msg_type = r.read_u8()? as u16;
            let msg_size = r.read_u16()? as usize;
            let msg_flags = r.read_u8()?;

            if track_creation_order {
                let _creation_order = r.read_u16()?;
            }

            // NIL type signals start of gap/padding to end of chunk.
            if msg_type == msg_types::NIL {
                break;
            }

            if msg_size == 0 {
                messages.push(HeaderMessage {
                    msg_type,
                    data: Bytes::new(),
                    flags: msg_flags,
                });
                continue;
            }

            let msg_data = r.slice_from_position(msg_size)?;
            r.skip(msg_size as u64);

            messages.push(HeaderMessage {
                msg_type,
                data: msg_data,
                flags: msg_flags,
            });
        }

        Ok(Self {
            version: 2,
            track_creation_order,
            messages,
        })
    }

    /// Find the first message of a given type.
    pub fn find_message(&self, msg_type: u16) -> Option<&HeaderMessage> {
        self.messages.iter().find(|m| m.msg_type == msg_type)
    }

    /// Find all messages of a given type.
    pub fn find_messages(&self, msg_type: u16) -> Vec<&HeaderMessage> {
        self.messages.iter().filter(|m| m.msg_type == msg_type).collect()
    }

    /// Check if a continuation message is present (meaning we'd need to
    /// fetch more data from a different file address).
    pub fn has_continuation(&self) -> bool {
        self.messages
            .iter()
            .any(|m| m.msg_type == msg_types::HEADER_CONTINUATION)
    }

    /// Extract continuation addresses from continuation messages.
    pub fn continuation_addresses(
        &self,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Vec<(u64, u64)>> {
        let mut continuations = Vec::new();
        for msg in &self.messages {
            if msg.msg_type == msg_types::HEADER_CONTINUATION {
                let mut r =
                    HDF5Reader::with_sizes(msg.data.clone(), size_of_offsets, size_of_lengths);
                let address = r.read_offset()?;
                let length = r.read_length()?;
                continuations.push((address, length));
            }
        }
        Ok(continuations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_v2_minimal() {
        let mut data = Vec::new();
        // OHDR signature
        data.extend_from_slice(&OHDR_SIGNATURE);
        // Version = 2
        data.push(2);
        // Flags = 0 (no timestamps, no creation order, 1-byte chunk size)
        data.push(0x00);
        // Chunk #0 size = 12 (4 bytes msg + 4 bytes padding + 4 bytes checksum)
        // We'll use a simple message: NIL type 0, size 0
        data.push(8); // chunk size: 4 bytes for a message + 4 for checksum

        // Message: type=DATASPACE(1), size=0, flags=0
        data.push(0x01); // type
        data.extend_from_slice(&0u16.to_le_bytes()); // size
        data.push(0x00); // flags

        // Checksum (4 bytes, just zeros for test)
        data.extend_from_slice(&0u32.to_le_bytes());

        let bytes = Bytes::from(data);
        let header = ObjectHeader::parse(&bytes, 8, 8).unwrap();
        assert_eq!(header.version, 2);
        assert_eq!(header.messages.len(), 1);
        assert_eq!(header.messages[0].msg_type, 1);
    }
}
