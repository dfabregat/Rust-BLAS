use num_complex::{Complex32,Complex64};

fn axpy<'a, T>(
    n: usize,
    alpha: T,
    x: &'a [T],
    incx: usize,
    y: &'a mut [T],
    incy: usize
)
where T: 'a + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy {
    if n == 0 {
        return
    }
    assert!(incx > 0);
    assert!(x.len() >= ((n - 1) * incx + 1));
    assert!(incy > 0);
    assert!(y.len() >= ((n - 1) * incy + 1));

    // Classic simple approach
    /*
    for i in 0..n {
        y[i*incy] = alpha * x[i*incx] + y[i*incy];
    }
    */

    // Iterator-based approach
    let iter_x = x.iter().step_by(incx).take(n);
    let iter_y = y.iter_mut().step_by(incy).take(n);
    for (x, y) in iter_x.zip(iter_y) {
        *y = alpha * *x + *y;
    }
}

/// # Description
///
/// The function `saxpy` computes
///
/// `y = alpha * x + y`
///
/// for scalar `alpha` and vectors `x` and `y`.
///
/// This is the `f32` (single precision floating point) version.
///
/// # Arguments
///
/// - `n`: number of elements of the vectors
/// - `alpha`: scaling factor
/// - `x`: vector x
/// - `incx`: step between consecutive elements of x
/// - `y`: vector y
/// - `incy`: step between consecutive elements of y
///
/// # Examples
///
/// ```
/// use blas::saxpy;
///
/// fn main() {
///     // Basic case
///     let n = 8;
///     let alpha = 2.0_f32;
///     let x = vec![1.0_f32; 8];
///     let incx = 1;
///     let mut y = vec![2.0_f32; 8];
///     let incy = 1;
///     let expected_y = vec![4.0_f32; 8];
///
///     saxpy(n, alpha, &x, incx, &mut y, incy);
///     assert_eq!(y, expected_y);
///
///     // With increment; notice the change in `n`
///     let n = 4;
///     let alpha = 2.0_f32;
///     let x = vec![1.0_f32; 8];
///     let incx = 2;
///     let mut y = vec![2.0_f32; 8];
///     let incy = 2;
///     let expected_y = vec![4.0_f32, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0];
///
///     saxpy(n, alpha, &x, incx, &mut y, incy);
///     assert_eq!(y, expected_y);
///
///     // With increment and an offset in the start of the vector
///     let n = 4;
///     let alpha = 2.0_f32;
///     let x = vec![1.0_f32; 8];
///     let incx = 2;
///     let mut y = vec![2.0_f32; 8];
///     let incy = 2;
///     let offset = 1;
///     let expected_y = vec![2.0_f32, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0];
///
///     saxpy(n, alpha, &x[offset..], incx, &mut y[offset..], incy);
///     assert_eq!(y, expected_y);
/// }
/// ```
/// # Panics
///
/// This function panics if:
/// - `incx == 0`, or
/// - `incy == 0`, or
/// - `x.len() < (n - 1) * incx + 1`, or
/// - `y.len() < (n - 1) * incy + 1`
///
pub fn saxpy(n: usize, alpha: f32, x: &[f32], incx: usize, y: &mut [f32], incy: usize) {
    axpy::<f32>(n, alpha, x, incx, y, incy)
}

pub fn daxpy(n: usize, alpha: f64, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
    axpy::<f64>(n, alpha, x, incx, y, incy)
}

pub fn caxpy(n: usize, alpha: Complex32, x: &[Complex32], incx: usize, y: &mut [Complex32], incy: usize) {
    axpy::<Complex32>(n, alpha, x, incx, y, incy)
}

pub fn zaxpy(n: usize, alpha: Complex64, x: &[Complex64], incx: usize, y: &mut [Complex64], incy: usize) {
    axpy::<Complex64>(n, alpha, x, incx, y, incy)
}

#[cfg(test)]
mod tests {
    use super::*;

    //
    // Extensive test cases, positive and negative, for the generic function
    //
    #[test]
    fn test_axpy_int_neg_alpha() {
        let n = 8;
        let alpha = -3;
        let x = vec![1; 8];
        let incx = 1;
        let mut y = vec![2; 8];
        let incy = 1;
        let expected_y = vec![-1; 8];
        axpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_axpy_int_n_0() {
        let n = 0;
        let alpha = 2;
        let x = vec![1; 8];
        let incx = 1;
        let mut y = vec![2; 8];
        let incy = 1;
        let expected_y = vec![2; 8];
        axpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_axpy_int_inc_1() {
        let n = 4;
        let alpha = 2;
        let x = vec![1; 8];
        let incx = 1;
        let mut y = vec![2; 8];
        let incy = 1;
        let expected_y = vec![4, 4, 4, 4, 2, 2, 2, 2];
        axpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_axpy_int_inc_1_offset() {
        let n = 4;
        let alpha = 2;
        let x = vec![1; 8];
        let incx = 1;
        let mut y = vec![2; 8];
        let incy = 1;
        let offset = 1;
        let expected_y = vec![2, 4, 4, 4, 4, 2, 2, 2];
        axpy(n, alpha, &x[offset..], incx, &mut y[offset..], incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_axpy_int_inc_n() {
        let n = 4;
        let alpha = 2;
        let x = vec![1; 8];
        let incx = 2;
        let mut y = vec![2; 8];
        let incy = 2;
        let expected_y = vec![4, 2, 4, 2, 4, 2, 4, 2];
        axpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_axpy_int_inc_n_offset() {
        let n = 4;
        let alpha = 2;
        let x = vec![1; 8];
        let incx = 2;
        let mut y = vec![2; 8];
        let incy = 2;
        let offset = 1;
        let expected_y = vec![2, 4, 2, 4, 2, 4, 2, 4];
        axpy(n, alpha, &x[offset..], incx, &mut y[offset..], incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_axpy_int_inc_mixed() {
        let n = 4;
        let alpha = 2;
        let x = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let incx = 1;
        let mut y = vec![5, 6, 7, 8, 1, 2, 3, 4];
        let incy = 2;
        let expected_y = vec![7, 6, 11, 8, 7, 2, 11, 4];
        axpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_axpy_int_large() {
        let n = 100_000;
        let alpha = 2;
        let x = vec![1; 100_000];
        let incx = 1;
        let mut y = vec![2; 100_000];
        let incy = 1;
        let expected_y = vec![4; 100_000];
        axpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    #[should_panic]
    fn test_axpy_int_incx_0() {
        let n = 8;
        let alpha = 2;
        let x = vec![1; 8];
        let incx = 0;
        let mut y = vec![2; 8];
        let incy = 1;
        axpy(n, alpha, &x, incx, &mut y, incy);
    }

    #[test]
    #[should_panic]
    fn test_axpy_int_incy_0() {
        let n = 8;
        let alpha = 2;
        let x = vec![1; 8];
        let incx = 1;
        let mut y = vec![2; 8];
        let incy = 0;
        axpy(n, alpha, &x, incx, &mut y, incy);
    }

    #[test]
    #[should_panic]
    fn test_axpy_int_out_of_bounds_n_x() {
        let n = 8;
        let alpha = 2;
        let x = vec![1; 7];
        let incx = 1;
        let mut y = vec![2; 8];
        let incy = 0;
        // would access x[7], off-by-one
        axpy(n, alpha, &x, incx, &mut y, incy);
    }

    #[test]
    #[should_panic]
    fn test_axpy_int_out_of_bounds_n_y() {
        let n = 8;
        let alpha = 2;
        let x = vec![1; 8];
        let incx = 1;
        let mut y = vec![2; 7];
        let incy = 0;
        // would access y[7], off-by-one
        axpy(n, alpha, &x, incx, &mut y, incy);
    }

    #[test]
    #[should_panic]
    fn test_axpy_int_out_of_bounds_incx() {
        let n = 4;
        let alpha = 2;
        let x = vec![1; 8];
        let incx = 3;
        let mut y = vec![2; 8];
        let incy = 1;
        // would access x[9]
        axpy(n, alpha, &x, incx, &mut y, incy);
    }

    #[test]
    #[should_panic]
    fn test_axpy_int_out_of_bounds_incy() {
        let n = 4;
        let alpha = 2;
        let x = vec![1; 8];
        let incx = 1;
        let mut y = vec![2; 8];
        let incy = 3;
        // would access y[9]
        axpy(n, alpha, &x, incx, &mut y, incy);
    }

    #[test]
    #[should_panic]
    fn test_axpy_int_out_of_bounds_offset() {
        let n = 4;
        let alpha = 2;
        let x = vec![1; 8];
        let incx = 1;
        let mut y = vec![2; 8];
        let incy = 1;
        let offset = 5;
        // would access x[8] and y[8], off-by-one
        axpy(n, alpha, &x[offset..], incx, &mut y[offset..], incy);
    }

    //
    // Fewer positive tests for each of the exposed "monomorphized" functions
    //

    //
    // saxpy
    //
    #[test]
    fn test_saxpy_inc_1_neg_alpha() {
        let n = 8;
        let alpha = -3f32;
        let x = vec![1f32; 8];
        let incx = 1;
        let mut y = vec![2f32; 8];
        let incy = 1;
        let expected_y = vec![-1f32; 8];
        saxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_saxpy_inc_n_offset() {
        let n = 4;
        let alpha = 2f32;
        let x = vec![1f32; 8];
        let incx = 2;
        let mut y = vec![2f32; 8];
        let incy = 2;
        let offset = 1;
        let expected_y = vec![2f32, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0];
        saxpy(n, alpha, &x[offset..], incx, &mut y[offset..], incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_saxpy_n_0() {
        let n = 0;
        let alpha = 2f32;
        let x = vec![1f32; 8];
        let incx = 1;
        let mut y = vec![2f32; 8];
        let incy = 1;
        let expected_y = vec![2f32; 8];
        saxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_saxpy_int_large() {
        let n = 100_000;
        let alpha = 2f32;
        let x = vec![1f32; 100_000];
        let incx = 1;
        let mut y = vec![2f32; 100_000];
        let incy = 1;
        let expected_y = vec![4f32; 100_000];
        saxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    //
    // daxpy
    //
    #[test]
    fn test_daxpy_inc_1_neg_alpha() {
        let n = 8;
        let alpha = -3f64;
        let x = vec![1f64; 8];
        let incx = 1;
        let mut y = vec![2f64; 8];
        let incy = 1;
        let expected_y = vec![-1f64; 8];
        daxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_daxpy_inc_n_offset() {
        let n = 4;
        let alpha = 2f64;
        let x = vec![1f64; 8];
        let incx = 2;
        let mut y = vec![2f64; 8];
        let incy = 2;
        let offset = 1;
        let expected_y = vec![2f64, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0];
        daxpy(n, alpha, &x[offset..], incx, &mut y[offset..], incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_daxpy_n_0() {
        let n = 0;
        let alpha = 2f64;
        let x = vec![1f64; 8];
        let incx = 1;
        let mut y = vec![2f64; 8];
        let incy = 1;
        let expected_y = vec![2f64; 8];
        daxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_daxpy_int_large() {
        let n = 100_000;
        let alpha = 2f64;
        let x = vec![1f64; 100_000];
        let incx = 1;
        let mut y = vec![2f64; 100_000];
        let incy = 1;
        let expected_y = vec![4f64; 100_000];
        daxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    //
    // caxpy
    //
    #[test]
    fn test_caxpy_inc_1() {
        const C32_1_1: Complex32 = Complex32::new(1.0,  1.0);
        const C32_2_2: Complex32 = Complex32::new(2.0, -2.0);
        const C32_3_3: Complex32 = Complex32::new(3.0,  3.0);
        // C32_4_0 = C32_1_1 * C32_2_2 + C32_3_3
        const C32_7_3: Complex32 = Complex32::new(7.0, 3.0);

        let n = 8;
        let alpha = C32_2_2;
        let x = vec![C32_1_1; 8];
        let incx = 1;
        let mut y = vec![C32_3_3; 8];
        let incy = 1;
        let expected_y = vec![C32_7_3; 8];
        caxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_caxpy_inc_n_offset() {
        const C32_1_1: Complex32 = Complex32::new(1.0, 1.0);
        const C32_2_2: Complex32 = Complex32::new(2.0, 2.0);
        const C32_3_3: Complex32 = Complex32::new(3.0, 3.0);
        // C32_3_7 = C32_1_1 * C32_2_2 + C32_3_3
        const C32_3_7: Complex32 = Complex32::new(3.0, 7.0);

        let n = 4;
        let alpha = C32_2_2;
        let x = vec![C32_1_1; 8];
        let incx = 2;
        let mut y = vec![C32_3_3; 8];
        let incy = 2;
        let expected_y = vec![C32_3_7, C32_3_3, C32_3_7, C32_3_3,
                              C32_3_7, C32_3_3, C32_3_7, C32_3_3];
        caxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_caxpy_n_0() {
        const C32_1_1: Complex32 = Complex32::new(1.0, 1.0);
        const C32_2_2: Complex32 = Complex32::new(2.0, 2.0);
        const C32_3_3: Complex32 = Complex32::new(3.0, 3.0);

        let n = 0;
        let alpha = C32_2_2;
        let x = vec![C32_1_1; 8];
        let incx = 1;
        let mut y = vec![C32_3_3; 8];
        let incy = 1;
        let expected_y = vec![C32_3_3; 8];
        caxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_caxpy_large() {
        const C32_1_1: Complex32 = Complex32::new(1.0, 1.0);
        const C32_2_2: Complex32 = Complex32::new(2.0, 2.0);
        const C32_3_3: Complex32 = Complex32::new(3.0, 3.0);
        // C32_3_7 = C32_1_1 * C32_2_2 + C32_3_3
        const C32_3_7: Complex32 = Complex32::new(3.0, 7.0);

        let n = 100_000;
        let alpha = C32_2_2;
        let x = vec![C32_1_1; 100_000];
        let incx = 1;
        let mut y = vec![C32_3_3; 100_000];
        let incy = 1;
        let expected_y = vec![C32_3_7; 100_000];
        caxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    //
    // zaxpy
    //
    #[test]
    fn test_zaxpy_inc_1() {
        const C64_1_1: Complex64 = Complex64::new(1.0,  1.0);
        const C64_2_2: Complex64 = Complex64::new(2.0, -2.0);
        const C64_3_3: Complex64 = Complex64::new(3.0,  3.0);
        // C64_4_0 = C64_1_1 * C64_2_2 + C64_3_3
        const C64_7_3: Complex64 = Complex64::new(7.0,  3.0);

        let n = 8;
        let alpha = C64_2_2;
        let x = vec![C64_1_1; 8];
        let incx = 1;
        let mut y = vec![C64_3_3; 8];
        let incy = 1;
        let expected_y = vec![C64_7_3; 8];
        zaxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_zaxpy_inc_n_offset() {
        const C64_1_1: Complex64 = Complex64::new(1.0, 1.0);
        const C64_2_2: Complex64 = Complex64::new(2.0, 2.0);
        const C64_3_3: Complex64 = Complex64::new(3.0, 3.0);
        // C64_3_7 = C64_1_1 * C64_2_2 + C64_3_3
        const C64_3_7: Complex64 = Complex64::new(3.0, 7.0);

        let n = 4;
        let alpha = C64_2_2;
        let x = vec![C64_1_1; 8];
        let incx = 2;
        let mut y = vec![C64_3_3; 8];
        let incy = 2;
        let expected_y = vec![C64_3_7, C64_3_3, C64_3_7, C64_3_3,
                              C64_3_7, C64_3_3, C64_3_7, C64_3_3];
        zaxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_zaxpy_n_0() {
        const C64_1_1: Complex64 = Complex64::new(1.0, 1.0);
        const C64_2_2: Complex64 = Complex64::new(2.0, 2.0);
        const C64_3_3: Complex64 = Complex64::new(3.0, 3.0);

        let n = 0;
        let alpha = C64_2_2;
        let x = vec![C64_1_1; 8];
        let incx = 1;
        let mut y = vec![C64_3_3; 8];
        let incy = 1;
        let expected_y = vec![C64_3_3; 8];
        zaxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }

    #[test]
    fn test_zaxpy_large() {
        const C64_1_1: Complex64 = Complex64::new(1.0, 1.0);
        const C64_2_2: Complex64 = Complex64::new(2.0, 2.0);
        const C64_3_3: Complex64 = Complex64::new(3.0, 3.0);
        // C64_3_7 = C64_1_1 * C64_2_2 + C64_3_3
        const C64_3_7: Complex64 = Complex64::new(3.0, 7.0);

        let n = 100_000;
        let alpha = C64_2_2;
        let x = vec![C64_1_1; 100_000];
        let incx = 1;
        let mut y = vec![C64_3_3; 100_000];
        let incy = 1;
        let expected_y = vec![C64_3_7; 100_000];
        zaxpy(n, alpha, &x, incx, &mut y, incy);
        assert_eq!(y, expected_y);
    }
}
