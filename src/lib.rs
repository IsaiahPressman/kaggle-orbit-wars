use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

#[pyfunction]
fn hello_from_rust() -> String {
    "Hello from owl!".to_string()
}

#[pyfunction]
fn hello_numpy(py: Python<'_>) -> PyResult<Bound<'_, PyArray2<f32>>> {
    let mut arr = Array2::<f32>::zeros((4, 2));
    arr[[0, 0]] = 1.;
    arr[[3, 1]] = 2.;
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
fn assert_release_build() {
    debug_assert!(false, "Running debug build")
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_rust, m)?)?;
    m.add_function(wrap_pyfunction!(hello_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(assert_release_build, m)?)?;
    Ok(())
}
