#![doc = "Asynchronous HDF5 metadata reader."]
#![warn(missing_docs)]

/// Chunk index types for mapping chunk coordinates to file byte ranges.
pub mod chunk_index;
/// HDF5 dataset — typed, shaped, chunked/contiguous array metadata.
pub mod dataset;
/// Endian-aware binary reader with HDF5 variable-width field support.
pub mod endian;
/// Error types.
pub mod error;
/// HDF5 file entry point.
pub mod file;
/// HDF5 group navigation.
pub mod group;
/// Object header parsing.
pub mod object_header;
/// Reader abstraction for async I/O.
pub mod reader;
/// Superblock parsing.
pub mod superblock;

/// HDF5 header message parsers.
pub mod messages;

/// B-tree implementations (v1 and v2).
pub mod btree;
/// Extensible Array chunk index reader.
pub mod extensible_array;
/// Fixed Array chunk index reader.
pub mod fixed_array;
/// Heap implementations (local and fractal).
pub mod heap;

// Re-exports for convenience.
pub use chunk_index::{ChunkIndex, ChunkLocation};
pub use dataset::HDF5Dataset;
pub use error::{HDF5Error, Result};
pub use file::HDF5File;
pub use group::HDF5Group;
pub use messages::attribute::{Attribute, AttributeValue};
pub use messages::data_layout::StorageLayout;
pub use messages::datatype::DataType;
pub use messages::filter_pipeline::FilterPipeline;
pub use object_header::ObjectHeader;
pub use reader::AsyncFileReader;
pub use superblock::Superblock;
