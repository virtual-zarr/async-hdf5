# async-hdf5

A pure-Rust, async HDF5 metadata reader. No libhdf5 dependency.

Designed for cloud-native workflows where you need to read HDF5 file structure and chunk locations over the network (S3, GCS, Azure, HTTP) without downloading entire files.

## Features

- **Async I/O** — all reads go through an `AsyncFileReader` trait, with built-in implementations for `object_store`, `reqwest`, and `tokio::fs`
- **Block caching** — coalesces scattered metadata reads into aligned 8 MiB block fetches, dramatically reducing request count for remote files
- **No C dependencies** — pure Rust binary parser, no libhdf5 required
- **Broad format support** — superblock versions 0-3, object header v1/v2, B-tree v1/v2, fractal heaps, fixed arrays
- **Chunk index extraction** — maps every chunk in a dataset to `(byte_offset, byte_length)` for building virtual Zarr stores

## Usage

```rust
use async_hdf5::{HDF5File, AsyncFileReader};
use object_store::local::LocalFileSystem;

let store = LocalFileSystem::new();
let path = object_store::path::Path::from("data.h5");
let reader = async_hdf5::reader::ObjectReader::new(store, path);

let file = HDF5File::open(reader).await?;
let root = file.root_group().await?;

// Navigate groups
let group = root.group("measurements").await?;
let dataset = group.dataset("temperature").await?;

// Inspect metadata
println!("shape: {:?}", dataset.shape());
println!("dtype: {:?}", dataset.dtype());
println!("chunk_shape: {:?}", dataset.chunk_shape());
println!("filters: {:?}", dataset.filters());

// Extract chunk byte ranges
let chunk_index = dataset.chunk_index().await?;
for chunk in chunk_index.iter() {
    println!(
        "chunk {:?} at offset {} ({} bytes)",
        chunk.indices, chunk.byte_offset, chunk.byte_length
    );
}
```

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `object_store` | yes | `ObjectReader` for S3/GCS/Azure/local via the `object_store` crate |
| `reqwest` | yes | `ReqwestReader` for HTTP range requests |
| `tokio` | yes | `TokioReader` for local async file I/O |

Disable defaults and pick only what you need:

```toml
[dependencies]
async-hdf5 = { version = "0.1", default-features = false, features = ["object_store"] }
```

## Custom readers

Implement `AsyncFileReader` to bring your own I/O backend:

```rust
#[async_trait]
pub trait AsyncFileReader: Send + Sync {
    async fn read_bytes(&self, start: u64, len: u64) -> Result<Bytes>;
    async fn file_size(&self) -> Result<u64>;
}
```

Wrap any reader in `BlockCache` for automatic aligned block caching:

```rust
use async_hdf5::reader::BlockCache;

let cached = BlockCache::new(my_reader);
let file = HDF5File::open(cached).await?;
```

## HDF5 coverage

| Category | Support |
|----------|---------|
| Superblock versions | 0, 1, 2, 3 |
| Object headers | v1, v2 (with continuation chains) |
| Group storage | v1 symbol table, v2 inline links, v2 dense (fractal heap + B-tree v2) |
| Chunk indexing | B-tree v1, B-tree v2, fixed array, single chunk |
| Storage layouts | chunked, contiguous, compact |
| Data types | fixed-point, floating-point, string, compound, variable-length, array, enum, opaque, bitfield, time |
| Filters | deflate, shuffle, fletcher32, zstd, and others (parsed but not applied) |
| Attributes | scalar, string, variable-length string (via global heap) |

Note: this crate reads **metadata only** — it does not decompress or decode array data. It is intended for use cases like building virtual Zarr stores where you need chunk locations but read the actual data through a separate path.

## License

Apache-2.0
