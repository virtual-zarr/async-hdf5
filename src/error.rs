use thiserror::Error;

/// Result type for async-hdf5 operations.
pub type Result<T> = std::result::Result<T, HDF5Error>;

/// Errors that can occur when reading HDF5 files.
#[derive(Debug, Error)]
pub enum HDF5Error {
    /// The file does not have a valid HDF5 signature.
    #[error("Not an HDF5 file: {hint}")]
    InvalidSignature {
        /// Byte offset that was checked.
        offset: u64,
        /// Human-readable hint about what was found instead.
        hint: String,
    },

    /// The superblock version is not supported.
    #[error("Unsupported superblock version: {0}")]
    UnsupportedSuperblockVersion(u8),

    /// The object header version is not supported.
    #[error("Unsupported object header version: {0}")]
    UnsupportedObjectHeaderVersion(u8),

    /// The data layout message version is not supported.
    #[error("Unsupported data layout version: {0}")]
    UnsupportedDataLayoutVersion(u8),

    /// The datatype class is not supported.
    #[error("Unsupported datatype class: {0}")]
    UnsupportedDatatypeClass(u8),

    /// The filter pipeline version is not supported.
    #[error("Unsupported filter pipeline version: {0}")]
    UnsupportedFilterPipelineVersion(u8),

    /// The chunk indexing type is not supported.
    #[error("Unsupported chunk indexing type: {0}")]
    UnsupportedChunkIndexingType(u8),

    /// The B-tree version is not supported.
    #[error("Unsupported B-tree version: {0}")]
    UnsupportedBTreeVersion(u8),

    /// The B-tree signature does not match what was expected.
    #[error("Invalid B-tree signature: expected {expected}, got {got}")]
    InvalidBTreeSignature {
        /// Expected signature string.
        expected: String,
        /// Actual signature found.
        got: String,
    },

    /// The heap version is not supported.
    #[error("Unsupported heap version: {0}")]
    UnsupportedHeapVersion(u8),

    /// The heap signature does not match what was expected.
    #[error("Invalid heap signature: expected {expected}, got {got}")]
    InvalidHeapSignature {
        /// Expected signature string.
        expected: String,
        /// Actual signature found.
        got: String,
    },

    /// A group member was not found at the given path.
    #[error("Group member not found: {0}")]
    NotFound(String),

    /// The path does not point to a group.
    #[error("Expected group at path: {0}")]
    NotAGroup(String),

    /// The path does not point to a dataset.
    #[error("Expected dataset at path: {0}")]
    NotADataset(String),

    /// The data ended before the expected number of bytes could be read.
    #[error("Unexpected end of data: needed {needed} bytes, had {available}")]
    UnexpectedEof {
        /// Number of bytes needed.
        needed: usize,
        /// Number of bytes available.
        available: usize,
    },

    /// An undefined address was encountered (unallocated storage).
    #[error("Undefined address encountered (unallocated storage)")]
    UndefinedAddress,

    /// The object header signature is invalid.
    #[error("Invalid object header signature: expected OHDR")]
    InvalidObjectHeaderSignature,

    /// The link type is not supported.
    #[error("Unsupported link type: {0}")]
    UnsupportedLinkType(u8),

    /// The message type is not supported.
    #[error("Unsupported message type: {0:#06x}")]
    UnsupportedMessageType(u16),

    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A general error with a descriptive message.
    #[error("{0}")]
    General(String),

    /// An object store error occurred.
    #[cfg(feature = "object_store")]
    #[error("Object store error: {0}")]
    ObjectStore(#[from] object_store::Error),

    /// An HTTP request error occurred.
    #[cfg(feature = "reqwest")]
    #[error("HTTP error: {0}")]
    Reqwest(#[from] reqwest::Error),
}
