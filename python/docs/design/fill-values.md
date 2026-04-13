# Fill Value Handling

This document describes how async-hdf5 handles fill values when mapping HDF5 metadata to the Zarr data model, and the design decisions behind the approach.

## The problem

HDF5 datasets carry fill value information in two places:

| Source | Location | Type |
|--------|----------|------|
| Header fill value | Dataset object header (message type 0x0005) | Typed bytes matching the dataset dtype |
| `_FillValue` attribute | Dataset attribute (CF convention) | May differ from the dataset dtype |
| `missing_value` attribute | Dataset attribute (older CF convention) | May differ from the dataset dtype |

Meanwhile, the Zarr data model has a single, typed `fill_value` field on each array, and CF-aware readers like xarray expect `_FillValue` attributes to be encoded in a specific format.

## Two distinct fill value concepts

The Zarr and CF data models use "fill value" for two different purposes:

**Zarr `fill_value`** (storage-level)
:   The default value returned for uninitialized or missing chunks. Set in `ArrayV3Metadata.fill_value`. Zarr handles its own serialization.

**CF `_FillValue` attribute** (data-level)
:   A sentinel value that CF-aware readers (xarray) use to mask individual data points as missing within chunks that _do_ contain data. Stored as an array attribute and must be encoded for xarray's `FillValueCoder`.

In many HDF5 files (especially those following CF conventions), the header fill value and the `_FillValue` attribute are the same value. But they can differ — the header fill value is often just the dtype default (e.g., `0.0` for float32), while `_FillValue` is an intentional sentinel like `-9999`.

## The approach

The `_consolidate_fill_value` function in `zarr.py` handles this mapping. It runs after dataset attributes are loaded and before `ArrayV3Metadata` is constructed.

### Zarr fill_value: from the HDF5 header

The Zarr `fill_value` comes from the HDF5 dataset header, decoded by `_decode_fill_value()`. This is the value zarr returns for uninitialized chunks and must match what the HDF5 library would return for unallocated space. If no header fill value is defined, the dtype default is used (e.g., `0.0` for float32).

The `_FillValue` attribute does **not** override the Zarr fill_value. Using the attribute value would cause zarr to return incorrect data for files where the header fill value and `_FillValue` attribute differ — a common case when the header has the dtype default and `_FillValue` is an explicit sentinel like `-9999`.

### _FillValue attribute: encoded for xarray

The `_FillValue` attribute is encoded via `FillValueCoder.encode()` so that xarray's Zarr backend can decode it. The encoding depends on the dtype:

| dtype kind | Encoding format |
|------------|----------------|
| `f` (float) | base64-encoded little-endian double |
| `c` (complex) | list of two base64-encoded little-endian doubles |
| `iu` (integer) | Python `int` |
| `b` (boolean) | Python `bool` |
| `U` (unicode string) | Python `str` |
| `S` (byte string) | base64-encoded bytes |

Before encoding, the attribute value is cast to the dataset's numpy dtype via `dtype.type(v).item()`. This ensures the encoded value matches the array's storage type.

### missing_value attribute: plain numeric

The `missing_value` attribute is passed through as a plain Python numeric (via `.item()` if it's a numpy scalar). xarray does not decode this attribute through `FillValueCoder` — it expects a native numeric type.

### Out-of-range fill values

Some files have `_FillValue` attributes that cannot be represented in the dataset's dtype (e.g., `_FillValue=-9999` on a `uint8` dataset). This typically happens when data was converted between formats and the fill value was preserved from the original dtype. In this case, async-hdf5 issues a warning and drops the `_FillValue` attribute. The data is still readable, but xarray will not mask fill values.

## Downstream reader compatibility

### xarray

xarray recognizes `_FillValue` and `missing_value` as CF masking attributes. Its Zarr backend decodes `_FillValue` via `FillValueCoder` but passes `missing_value` through as-is. With `mask_and_scale=True` (the default), xarray replaces values equal to `_FillValue` or `missing_value` with `NaN`.

The `use_zarr_fill_value_as_mask` option controls whether the Zarr-level `fill_value` is also treated as a masking sentinel. When the header fill value and `_FillValue` attribute are the same (the common case), both paths produce correct results.

## Example: ICESat-2 ATL06

ATL06 datasets use `3.4028235e+38` (float32 max) as a fill value sentinel. The HDF5 structure looks like:

```
h_li:
  header fill_value: 3.4028235e+38 (from dataset creation property)
  _FillValue attribute: 3.4028235e+38 (CF convention)
  dtype: float32
  filters: [shuffle, deflate]
```

async-hdf5 maps this to:

```
zarr fill_value: 3.4028235e+38      (from HDF5 header)
_FillValue attr: "AAAAAn9/f0A="     (base64-encoded for xarray)
```

When xarray opens this with `mask_and_scale=True`, values equal to `3.4028235e+38` are replaced with `NaN`, which is the expected behavior for CF-compliant data.
