# async-hdf5 (Python)

> **Warning — Experimental.**
> This library is under active development and not ready for production use.
> The API may change without notice. Known limitations:
>
> - **Metadata only** — does not decompress or decode array data; designed for serving HDF5 via Zarr's data protocol.
> - **Incomplete HDF5 coverage** — object header v0, HDF5 Time datatype (class 2), virtual dataset layout, and external data links are not supported. Some compound/array dtype edge cases produce incorrect numpy dtype mappings.
> - **Limited testing on real-world files** — validated against the HDF5 library test suite (59% pass rate), GDAL autotest files, and a small set of NASA/NOAA data. Many exotic HDF5 features remain untested.
> - **No fuzz testing** — the binary parser has not been fuzz-tested against adversarial inputs. While known panics have been fixed, corrupt files may trigger unexpected errors.
> - **Sparse array performance** — fixed array chunk indexing reads the entire dense index into memory, which can be expensive for very large, mostly-empty datasets.

Python bindings for the [`async-hdf5`](../README.md) Rust crate. Read HDF5 file metadata asynchronously from local disk or cloud storage (S3, GCS, Azure) without libhdf5.

## Install

```bash
pip install async-hdf5
```

Requires Python 3.11+.

## Usage

```python exec="on" session="demo"
import asyncio
import tempfile
import pathlib

import h5py
import numpy as np

# Create a sample HDF5 file
tmpdir = tempfile.mkdtemp()
filepath = str(pathlib.Path(tmpdir) / "sample.h5")

rng = np.random.default_rng(42)
with h5py.File(filepath, "w") as f:
    f.attrs["Conventions"] = "CF-1.8"
    f.attrs["title"] = "Sample temperature dataset"

    f.create_dataset("time", data=np.arange(12, dtype=np.float64))
    f["time"].attrs["units"] = "months since 2020-01-01"

    f.create_dataset("lat", data=np.linspace(-90, 90, 18, dtype=np.float64))
    f["lat"].attrs["units"] = "degrees_north"

    f.create_dataset("lon", data=np.linspace(-180, 180, 36, dtype=np.float64))
    f["lon"].attrs["units"] = "degrees_east"

    f.create_dataset(
        "temperature",
        data=rng.standard_normal((12, 18, 36), dtype=np.float32) * 10 + 280,
        chunks=(4, 18, 36),
        compression="gzip",
    )
    f["temperature"].attrs["units"] = "K"
    f["temperature"].attrs["long_name"] = "Near-Surface Air Temperature"
```

### Opening an HDF5 file

Any object implementing the [obspec](https://github.com/developmentseed/obspec) `GetRangeAsync` and `GetRangesAsync` protocols works as a store — including all [obstore](https://github.com/developmentseed/obstore) backends (S3, GCS, Azure, local, HTTP) and those compiled as part of async_hdf5.

```python exec="on" source="above" result="code" session="demo"
import asyncio

from async_hdf5.store import LocalStore
from async_hdf5 import HDF5File

store = LocalStore()


async def inspect():
    file = await HDF5File.open(filepath, store=store)
    root = await file.root_group()

    # Group attributes
    attrs = await root.attributes()
    print(f"Title: {attrs['title']}")
    print(f"Children: {await root.children()}")

    # Dataset metadata
    ds = await root.dataset("temperature")
    print(f"\nShape: {ds.shape}")
    print(f"Dtype: {ds.numpy_dtype}")
    print(f"Chunk shape: {ds.chunk_shape}")
    print(f"Filters: {ds.filters}")

    # Chunk index — maps grid coordinates to byte ranges
    chunk_index = await ds.chunk_index()
    print(f"\nGrid shape: {chunk_index.grid_shape}")
    print(f"Number of chunks: {len(chunk_index)}")
    for loc in chunk_index:
        print(
            f"  Chunk {loc.indices}: offset={loc.byte_offset}, length={loc.byte_length}"
        )


asyncio.run(inspect())
```

### xarray backend

Open any HDF5 file as an xarray Dataset using the `async_hdf5` engine:

```python exec="on" source="above" result="code" session="demo"
import xarray as xr

ds = xr.open_dataset(filepath, engine="async_hdf5")
print(ds)
print()
print(ds["temperature"])
```

For cloud storage, pass an ObjectStore:

```python
from async_hdf5.store import S3Store

s3 = S3Store(bucket="noaa-goes16", region="us-east-1", skip_signature=True)
ds = xr.open_dataset(
    "ABI-L2-MCMIPF/2024/099/18/file.nc",
    engine="async_hdf5",
    store=s3,
)
```

### Zarr store

Under the hood, the xarray backend uses `open_hdf5` which returns an `HDF5Store` — a read-only Zarr v3 store backed by async-hdf5. You can also use it directly:

```python exec="on" source="above" result="code" session="demo"
from async_hdf5.zarr import open_hdf5

zarr_store = asyncio.run(open_hdf5(path=filepath, store=store))
ds = xr.open_dataset(zarr_store, engine="zarr", consolidated=False, zarr_format=3)
print(ds)
```

### VirtualiZarr integration

`async_hdf5.virtualizarr` returns a `ManifestStore` containing virtual chunk references. No array data is read — only metadata and byte offsets:

```python exec="on" source="above" result="code" session="demo"
from async_hdf5.virtualizarr import open_virtual_hdf5

manifest_store = asyncio.run(
    open_virtual_hdf5(filepath, store=store, url=f"file://{filepath}")
)
vds = manifest_store.to_virtual_dataset()
print(vds)
```

```python exec="on" session="demo"
# Cleanup
import shutil

shutil.rmtree(tmpdir)
```

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
