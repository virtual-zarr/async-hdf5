use std::collections::HashMap;
use std::sync::Arc;

use async_hdf5::messages::datatype::{ByteOrder, DataType};
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;

use crate::chunk_index::PyChunkIndex;
use crate::group::attribute_value_to_py;

#[pyclass(name = "HDF5Dataset", frozen)]
pub(crate) struct PyHDF5Dataset {
    inner: Arc<async_hdf5::HDF5Dataset>,
}

impl PyHDF5Dataset {
    pub(crate) fn new(ds: async_hdf5::HDF5Dataset) -> Self {
        Self {
            inner: Arc::new(ds),
        }
    }
}

/// Convert a ByteOrder to a numpy endian character.
fn endian_char(bo: &ByteOrder) -> &'static str {
    match bo {
        ByteOrder::LittleEndian => "<",
        ByteOrder::BigEndian => ">",
        ByteOrder::Vax => "<",
        ByteOrder::NotApplicable => "|",
    }
}

/// Convert a DataType to a numpy-compatible dtype string (e.g., "<f4", ">i8", "<c16").
///
/// Returns `Err` with a descriptive message for types that cannot be
/// meaningfully represented as a fixed numpy dtype (VarLen, Reference).
/// The error message suggests using `drop_variables` to skip the dataset.
fn datatype_to_numpy_str(dt: &DataType) -> Result<String, String> {
    match dt {
        DataType::FixedPoint {
            size,
            signed,
            byte_order,
            ..
        } => {
            let e = if *size == 1 {
                "|"
            } else {
                endian_char(byte_order)
            };
            let c = if *signed { "i" } else { "u" };
            Ok(format!("{}{}{}", e, c, size))
        }
        DataType::FloatingPoint {
            size, byte_order, ..
        } => Ok(format!("{}f{}", endian_char(byte_order), size)),
        DataType::String { size, .. } => Ok(format!("|S{}", size)),
        DataType::Enum { base_type, .. } => datatype_to_numpy_str(base_type),
        DataType::Bitfield {
            size, byte_order, ..
        } => {
            let e = if *size == 1 {
                "|"
            } else {
                endian_char(byte_order)
            };
            Ok(format!("{}u{}", e, size))
        }
        DataType::Compound { size, fields, .. } if fields.len() == 2 => {
            // Detect complex: compound with two float fields named r/i or real/imag
            let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
            let both_float = fields
                .iter()
                .all(|f| matches!(f.dtype, DataType::FloatingPoint { .. }));
            if both_float && (names == ["r", "i"] || names == ["real", "imag"]) {
                let bo = match &fields[0].dtype {
                    DataType::FloatingPoint { byte_order, .. } => endian_char(byte_order),
                    _ => "|",
                };
                Ok(format!("{}c{}", bo, size))
            } else {
                Ok(format!("|V{}", size))
            }
        }
        DataType::VarLen { is_string, .. } => {
            if *is_string {
                Err(
                    "Variable-length string datatype cannot be represented as a fixed \
                     numpy dtype. Use drop_variables to skip this dataset."
                        .into(),
                )
            } else {
                Err(
                    "Variable-length sequence datatype cannot be represented as a fixed \
                     numpy dtype. Use drop_variables to skip this dataset."
                        .into(),
                )
            }
        }
        DataType::Reference { ref_type, .. } => Err(format!(
            "HDF5 reference datatype (ref_type={}) is not supported. \
             Use drop_variables to skip this dataset.",
            ref_type
        )),
        // Opaque, Compound (non-complex), Array → void bytes
        _ => {
            let size = dt.size();
            if size > 1_048_576 {
                Err(format!(
                    "Datatype too large for numpy void dtype: {} bytes. \
                     Use drop_variables to skip this dataset.",
                    size
                ))
            } else {
                Ok(format!("|V{}", size))
            }
        }
    }
}

#[pymethods]
impl PyHDF5Dataset {
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    #[getter]
    fn shape(&self) -> Vec<u64> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn dtype(&self) -> String {
        format!("{:?}", self.inner.dtype())
    }

    /// Numpy-compatible dtype string (e.g., "<f4", ">i8", "<c16").
    ///
    /// Raises ``ValueError`` for datatypes that cannot be represented as
    /// a fixed numpy dtype (variable-length, reference).
    #[getter]
    fn numpy_dtype(&self) -> PyResult<String> {
        datatype_to_numpy_str(self.inner.dtype()).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[getter]
    fn element_size(&self) -> u32 {
        self.inner.element_size()
    }

    #[getter]
    fn chunk_shape(&self) -> Option<Vec<u64>> {
        self.inner.chunk_shape().map(|s| s.to_vec())
    }

    /// Filter pipeline as a list of dicts: [{"id": int, "name": str, "client_data": [int]}].
    #[getter]
    fn filters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let pipeline = self.inner.filters();
        let list = pyo3::types::PyList::empty(py);
        for f in &pipeline.filters {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", f.id)?;
            dict.set_item("name", f.display_name())?;
            dict.set_item("client_data", f.client_data.to_vec().into_pyobject(py)?)?;
            list.append(dict)?;
        }
        Ok(list.into_any().unbind())
    }

    /// Raw fill value bytes, or None if not set.
    #[getter]
    fn fill_value(&self) -> Option<Vec<u8>> {
        self.inner.fill_value().map(|v| v.to_vec())
    }

    /// Whether this dataset has a null dataspace (no data).
    #[getter]
    fn is_null_dataspace(&self) -> bool {
        self.inner.is_null_dataspace()
    }

    /// Whether this dataset references external data files.
    #[getter]
    fn has_external_storage(&self) -> bool {
        self.inner.has_external_storage()
    }

    fn chunk_index<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ds = self.inner.clone();
        future_into_py(py, async move {
            let index = ds
                .chunk_index()
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyChunkIndex::new(index.clone()))
        })
    }

    /// Fetch multiple chunks in a single batched I/O call.
    ///
    /// Takes a list of chunk index tuples and returns a list of
    /// ``bytes | None`` in the same order.
    fn batch_get_chunks<'py>(
        &'py self,
        py: Python<'py>,
        chunk_indices: Vec<Vec<u64>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ds = self.inner.clone();
        future_into_py(py, async move {
            let results = ds
                .batch_get_chunks(&chunk_indices)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(results
                .into_iter()
                .map(|r| r.map(|b| b.to_vec()))
                .collect::<Vec<_>>())
        })
    }

    /// Fetch multiple byte ranges in a single batched I/O call.
    ///
    /// Takes a list of ``(offset, length)`` tuples and returns a list of
    /// ``bytes`` in the same order.  No chunk index lookup is performed —
    /// the caller must supply pre-resolved byte ranges.
    fn batch_fetch_ranges<'py>(
        &'py self,
        py: Python<'py>,
        ranges: Vec<(u64, u64)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ds = self.inner.clone();
        future_into_py(py, async move {
            let results = ds
                .batch_fetch_ranges(&ranges)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(results.into_iter().map(|b| b.to_vec()).collect::<Vec<_>>())
        })
    }

    fn attributes<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ds = self.inner.clone();
        future_into_py(py, async move {
            let attrs = ds.attributes().await;
            let dict: HashMap<String, Py<PyAny>> = Python::attach(|py| {
                attrs
                    .into_iter()
                    .map(|a| (a.name, attribute_value_to_py(py, a.value)))
                    .collect()
            });
            Ok(dict)
        })
    }
}
