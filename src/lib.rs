mod rl;
pub mod rules_engine;

use pyo3::prelude::*;

#[pyfunction]
fn assert_release_build() {
    debug_assert!(false, "Running debug build")
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(assert_release_build, m)?)?;
    rl::add_to_module(m)?;
    Ok(())
}
