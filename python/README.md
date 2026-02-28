# async-hdf5 (Python)

Python bindings for the [`async-hdf5`](../README.md) Rust crate. Read HDF5 file metadata asynchronously from local disk or cloud storage (S3, GCS, Azure) without libhdf5.

## Install

```bash
pip install async-hdf5
```

Requires Python 3.11+.

## Usage

```python
import obstore
from async_hdf5 import HDF5File

store = obstore.store.LocalStore()
file = await HDF5File.open("data.h5", store=store)
root = file.root_group()

# Navigate
group = await root.group("measurements")
ds = await group.dataset("temperature")

# Inspect
print(ds.shape)         # (1000, 500)
print(ds.numpy_dtype)   # <f4
print(ds.chunk_shape)   # (100, 100)
print(ds.filters)       # [{'id': 1, 'name': 'deflate', ...}]

# Chunk byte ranges
index = await ds.chunk_index()
for chunk in index:
    print(chunk.indices, chunk.byte_offset, chunk.byte_length)
```

### Cloud storage

```python
import obstore

store = obstore.store.S3Store(bucket="my-bucket", region="us-west-2")
file = await HDF5File.open("path/to/file.h5", store=store)
```

Any object implementing the [obspec](https://github.com/developmentseed/obspec) `GetRangeAsync` and `GetRangesAsync` protocols works as a store.

### VirtualiZarr integration

With `virtualizarr` installed, you can open HDF5 files as virtual xarray datasets:

```python
from async_hdf5 import open_virtual_hdf5
import obstore

store = obstore.store.S3Store(bucket="my-bucket", region="us-west-2")
ds = await open_virtual_hdf5(
    "path/to/file.h5",
    store=store,
    url="s3://my-bucket/path/to/file.h5",
)
# ds is an xarray.Dataset backed by ManifestStore — no data read yet
```

This extracts chunk manifests and converts HDF5 filters (deflate, shuffle, zstd) to Zarr v3 codecs automatically.

## API

### `HDF5File`

- `await HDF5File.open(path, *, store, block_size=8_388_608)` — open a file
- `file.root_group()` — get the root group

### `HDF5Group`

- `group.name` — group name
- `await group.group(name)` — child group by name
- `await group.dataset(name)` — child dataset by name
- `await group.navigate(path)` — navigate a `/`-separated path
- `await group.children()` — list `(name, type)` pairs
- `await group.group_names()` — list child group names
- `await group.dataset_names()` — list child dataset names
- `await group.attributes()` — `dict[str, Any]` of attributes

### `HDF5Dataset`

- `ds.name`, `ds.shape`, `ds.ndim`, `ds.dtype`, `ds.numpy_dtype`
- `ds.element_size`, `ds.chunk_shape`, `ds.filters`, `ds.fill_value`
- `await ds.chunk_index()` — get a `ChunkIndex`

### `ChunkIndex`

Iterable collection of `ChunkLocation` objects. Properties: `grid_shape`, `chunk_shape`, `dataset_shape`.

### `ChunkLocation`

- `chunk.indices` — tuple of chunk grid coordinates
- `chunk.byte_offset` — offset in file
- `chunk.byte_length` — size in bytes
- `chunk.filter_mask` — bitmask of applied filters

## Development

The Python package is built with [maturin](https://www.maturin.rs/) from the `python/` directory. [uv](https://docs.astral.sh/uv/) manages the Python environment and dependencies.

```bash
cd python
uv sync
```

### Develop mode (debug, fast compile)

Compiles the Rust extension in debug mode and installs it into the virtualenv. Fast iteration for development — no optimizations, includes debug symbols.

```bash
uv run maturin develop
```

### Release mode (optimized)

Compiles with `--release` (LTO + single codegen unit, as configured in `Cargo.toml`). Use this to benchmark or test production-like performance.

```bash
uv run maturin develop --release
```

### Running tests

```bash
uv run pytest
```

### Building a wheel

```bash
uv run maturin build --release
```

The wheel is written to `target/wheels/`.

## License

Apache-2.0
