extern crate num_complex;

pub mod blas1;

pub use blas1::{scal::{sscal, dscal, cscal, zscal}};
pub use blas1::{axpy::{saxpy, daxpy, caxpy, zaxpy}};
