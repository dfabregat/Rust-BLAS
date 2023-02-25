use num_complex::{Complex32,Complex64};

fn scal<'a, T>(
    n: usize,
    alpha: T,
    x: &'a mut [T],
    incx: usize
)
where T: 'a + std::ops::Mul<Output = T> + Copy {
    if n == 0 {
        return
    }
    assert!(incx > 0);
    assert!(x.len() >= ((n - 1) * incx + 1));

    for i in (0..x.len()).step_by(incx).take(n) {
        x[i] = alpha * x[i];
    }
}

/// # Description
///
/// The function `sscal` scales `n` elements of a vector `x`
/// by a constant `alpha`. Elements may separated by steps of
/// `incx`.
///
/// This is the `f32` (single precision floating point) version.
///
/// # Arguments
///
/// - `n`: number of elements to scale
/// - `alpha`: scaling factor
/// - `x`: vector to scale
/// - `incx`: step between consecutive elements to be scaled
///
/// # Examples
///
/// ```
/// use blas::sscal;
///
/// fn main() {
///     // Basic case
///     let mut x = vec![1.0_f32; 8];
///     let alpha = 2.0_f32;
///     let n = 8;
///     let incx = 1;
///     let expected_x = vec![2.0_f32; 8];
///
///     sscal(n, alpha, &mut x, incx);
///     assert_eq!(x, expected_x);
///
///     // With increment; notice the change in `n`
///     let mut x = vec![1.0_f32; 8];
///     let alpha = 2.0_f32;
///     let n = 4;
///     let incx = 2;
///     let expected_x = vec![2.0_f32, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0];
///
///     sscal(n, alpha, &mut x, incx);
///     assert_eq!(x, expected_x);
///
///     // With increment and an offset in the start of the vector
///     let mut x = vec![1.0_f32; 8];
///     let alpha = 2.0_f32;
///     let n = 4;
///     let incx = 2;
///     let offset = 1;
///     let expected_x = vec![1.0_f32, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
///
///     sscal(n, alpha, &mut x[offset..], incx);
///     assert_eq!(x, expected_x);
/// }
/// ```
/// # Panics
///
/// This function panics if:
/// - `incx == 0`, or
/// - `x.len() < (n - 1) * incx + 1`
///
pub fn sscal(n: usize, alpha: f32, x: &mut [f32], incx: usize) {
    scal::<f32>(n, alpha, x, incx)
}

/// # Description
///
/// The function `dscal` scales `n` elements of a vector `x`
/// by a constant `alpha`. Elements may separated by steps of
/// `incx`.
///
/// This is the `f64` (double precision floating point) version.
///
/// # Arguments
///
/// - `n`: number of elements to scale
/// - `alpha`: scaling factor
/// - `x`: vector to scale
/// - `incx`: step between consecutive elements to be scaled
///
/// # Examples
///
/// ```
/// use blas::dscal;
///
/// fn main() {
///     // Basic case
///     let mut x = vec![1.0_f64; 8];
///     let alpha = 2.0_f64;
///     let n = 8;
///     let incx = 1;
///     let expected_x = vec![2.0_f64; 8];
///
///     dscal(n, alpha, &mut x, incx);
///     assert_eq!(x, expected_x);
///
///     // With increment; notice the change in `n`
///     let mut x = vec![1.0_f64; 8];
///     let alpha = 2.0_f64;
///     let n = 4;
///     let incx = 2;
///     let expected_x = vec![2.0_f64, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0];
///
///     dscal(n, alpha, &mut x, incx);
///     assert_eq!(x, expected_x);
///
///     // With increment and an offset in the start of the vector
///     let mut x = vec![1.0_f64; 8];
///     let alpha = 2.0_f64;
///     let n = 4;
///     let incx = 2;
///     let offset = 1;
///     let expected_x = vec![1.0_f64, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
///
///     dscal(n, alpha, &mut x[offset..], incx);
///     assert_eq!(x, expected_x);
/// }
/// ```
/// # Panics
///
/// This function panics if:
/// - `incx == 0`, or
/// - `x.len() < (n - 1) * incx + 1`
///
pub fn dscal(n: usize, alpha: f64, x: &mut [f64], incx: usize) {
    scal::<f64>(n, alpha, x, incx)
}

/// # Description
///
/// The function `cscal` scales `n` elements of a vector `x`
/// by a constant `alpha`. Elements may separated by steps of
/// `incx`.
///
/// This is the `Complex32` (single precision complex) version.
///
/// # Arguments
///
/// - `n`: number of elements to scale
/// - `alpha`: scaling factor
/// - `x`: vector to scale
/// - `incx`: step between consecutive elements to be scaled
///
/// # Examples
///
/// ```
/// use blas::cscal;
/// use num_complex::Complex32;
///
/// fn main() {
///     const C32_1_1: Complex32 = Complex32::new(1.0, 1.0);
///     const C32_2_2: Complex32 = Complex32::new(2.0, 2.0);
///     // C32_0_4 = C32_1_1 * C32_2_2
///     const C32_0_4: Complex32 = Complex32::new(0.0, 4.0);
///
///     // Basic case
///     let mut x = vec![C32_1_1; 8];
///     let alpha = C32_2_2;
///     let n = 8;
///     let incx = 1;
///     let expected_x = vec![C32_0_4; 8];
///
///     cscal(n, alpha, &mut x, incx);
///     assert_eq!(x, expected_x);
///
///     // With increment; notice the change in `n`
///     let mut x = vec![C32_1_1; 8];
///     let alpha = C32_2_2;
///     let n = 4;
///     let incx = 2;
///     let expected_x = vec![C32_0_4, C32_1_1, C32_0_4, C32_1_1,
///                           C32_0_4, C32_1_1, C32_0_4, C32_1_1];
///
///     cscal(n, alpha, &mut x, incx);
///     assert_eq!(x, expected_x);
///
///     // With increment and an offset in the start of the vector
///     let mut x = vec![C32_1_1; 8];
///     let alpha = C32_2_2;
///     let n = 4;
///     let incx = 2;
///     let offset = 1;
///     let expected_x = vec![C32_1_1, C32_0_4, C32_1_1, C32_0_4,
///                           C32_1_1, C32_0_4, C32_1_1, C32_0_4];
///
///     cscal(n, alpha, &mut x[offset..], incx);
///     assert_eq!(x, expected_x);
/// }
/// ```
/// # Panics
///
/// This function panics if:
/// - `incx == 0`, or
/// - `x.len() < (n - 1) * incx + 1`
///
pub fn cscal(n: usize, alpha: Complex32, x: &mut [Complex32], incx: usize) {
    scal::<Complex32>(n, alpha, x, incx)
}

/// # Description
///
/// The function `zscal` scales `n` elements of a vector `x`
/// by a constant `alpha`. Elements may separated by steps of
/// `incx`.
///
/// This is the `Complex32` (single precision complex) version.
///
/// # Arguments
///
/// - `n`: number of elements to scale
/// - `alpha`: scaling factor
/// - `x`: vector to scale
/// - `incx`: step between consecutive elements to be scaled
///
/// # Examples
///
/// ```
/// use blas::zscal;
/// use num_complex::Complex64;
///
/// fn main() {
///     const C64_1_1: Complex64 = Complex64::new(1.0, 1.0);
///     const C64_2_2: Complex64 = Complex64::new(2.0, 2.0);
///     // C64_0_4 = C64_1_1 * C64_2_2
///     const C64_0_4: Complex64 = Complex64::new(0.0, 4.0);
///
///     // Basic case
///     let mut x = vec![C64_1_1; 8];
///     let alpha = C64_2_2;
///     let n = 8;
///     let incx = 1;
///     let expected_x = vec![C64_0_4; 8];
///
///     zscal(n, alpha, &mut x, incx);
///     assert_eq!(x, expected_x);
///
///     // With increment; notice the change in `n`
///     let mut x = vec![C64_1_1; 8];
///     let alpha = C64_2_2;
///     let n = 4;
///     let incx = 2;
///     let expected_x = vec![C64_0_4, C64_1_1, C64_0_4, C64_1_1,
///                           C64_0_4, C64_1_1, C64_0_4, C64_1_1];
///
///     zscal(n, alpha, &mut x, incx);
///     assert_eq!(x, expected_x);
///
///     // With increment and an offset in the start of the vector
///     let mut x = vec![C64_1_1; 8];
///     let alpha = C64_2_2;
///     let n = 4;
///     let incx = 2;
///     let offset = 1;
///     let expected_x = vec![C64_1_1, C64_0_4, C64_1_1, C64_0_4,
///                           C64_1_1, C64_0_4, C64_1_1, C64_0_4];
///
///     zscal(n, alpha, &mut x[offset..], incx);
///     assert_eq!(x, expected_x);
/// }
/// ```
/// # Panics
///
/// This function panics if:
/// - `incx == 0`, or
/// - `x.len() < (n - 1) * incx + 1`
///
pub fn zscal(n: usize, alpha: Complex64, x: &mut [Complex64], incx: usize) {
    scal::<Complex64>(n, alpha, x, incx)
}

#[cfg(test)]
mod tests {
    use super::*;

    //
    // Extensive test cases, positive and negative, for the generic function
    //
    #[test]
    fn test_scal_int_inc_1() {
        let mut x = vec![1; 8];
        let alpha = 2;
        let n = 4;
        let incx = 1;
        let expected_x = vec![2, 2, 2, 2, 1, 1, 1, 1];
        scal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_scal_int_inc_1_offset() {
        let mut x = vec![1; 8];
        let alpha = 2;
        let n = 4;
        let incx = 1;
        let offset = 1;
        let expected_x = vec![1, 2, 2, 2, 2, 1, 1, 1];
        scal(n, alpha, &mut x[offset..], incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_scal_int_inc_n() {
        let mut x = vec![1; 8];
        let alpha = 2;
        let n = 4;
        let incx = 2;
        let expected_x = vec![2, 1, 2, 1, 2, 1, 2, 1];
        scal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_scal_int_inc_n_offset() {
        let mut x = vec![1; 8];
        let alpha = 2;
        let n = 4;
        let incx = 2;
        let offset = 1;
        let expected_x = vec![1, 2, 1, 2, 1, 2, 1, 2];
        scal(n, alpha, &mut x[offset..], incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_scal_int_neg_alpha() {
        let mut x = vec![1; 8];
        let alpha = -2;
        let n = 8;
        let incx = 1;
        let expected_x = vec![-2; 8];
        scal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_scal_int_n_0() {
        let mut x = vec![1; 8];
        let alpha = 2;
        let n = 0;
        let incx = 1;
        let expected_x = vec![1; 8];
        scal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_scal_int_large() {
        let mut x = vec![1; 100_000];
        let alpha = 2;
        let n = 100_000;
        let incx = 1;
        let expected_x = vec![2; 100_000];
        scal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    #[should_panic]
    fn test_scal_int_inc_0() {
        let mut x = vec![1; 8];
        let alpha = 2;
        let n = 4;
        let incx = 0;
        let expected_x = vec![2, 1, 2, 1, 2, 1, 2, 1];
        scal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    #[should_panic]
    fn test_scal_int_out_of_bounds_n() {
        let mut x = vec![1; 8];
        let alpha = 2;
        let n = 9;
        let incx = 1;
        // would access x[9]
        scal(n, alpha, &mut x, incx);
    }

    #[test]
    #[should_panic]
    fn test_scal_int_out_of_bounds_incx() {
        let mut x = vec![1; 8];
        let alpha = 2;
        let n = 4;
        let incx = 3;
        // would access x[10]
        scal(n, alpha, &mut x, incx);
    }

    #[test]
    #[should_panic]
    fn test_scal_int_out_of_bounds_offset() {
        let mut x = vec![1; 8];
        let alpha = 2;
        let n = 4;
        let incx = 1;
        let offset = 5;
        // would access x[8]
        scal(n, alpha, &mut x[offset..], incx);
    }

    //
    // Fewer positive tests for each of the exposed "monomorphized" functions
    //

    //
    // sscal
    //
    #[test]
    fn test_sscal_inc_1_neg_alpha() {
        let mut x = vec![1f32; 8];
        let alpha = -2f32;
        let n = 8;
        let incx = 1;
        let expected_x = vec![-2f32; 8];
        sscal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_sscal_inc_n_offset() {
        let mut x = vec![1f32; 8];
        let alpha = 2f32;
        let n = 4;
        let incx = 2;
        let offset = 1;
        let expected_x = vec![1f32, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        sscal(n, alpha, &mut x[offset..], incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_sscal_n_0() {
        let mut x = vec![1f32; 8];
        let alpha = 2f32;
        let n = 0;
        let incx = 1;
        let expected_x = vec![1f32; 8];
        sscal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_sscal_int_large() {
        let mut x = vec![1f32; 100_000];
        let alpha = 2f32;
        let n = 100_000;
        let incx = 1;
        let expected_x = vec![2f32; 100_000];
        sscal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    //
    // dscal
    //
    #[test]
    fn test_dscal_inc_1_neg_alpha() {
        let mut x = vec![1f64; 8];
        let alpha = -2f64;
        let n = 8;
        let incx = 1;
        let expected_x = vec![-2f64; 8];
        dscal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_dscal_inc_n_offset() {
        let mut x = vec![1f64; 8];
        let alpha = 2f64;
        let n = 4;
        let incx = 2;
        let offset = 1;
        let expected_x = vec![1f64, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        dscal(n, alpha, &mut x[offset..], incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_dscal_n_0() {
        let mut x = vec![1f64; 8];
        let alpha = 2f64;
        let n = 0;
        let incx = 1;
        let expected_x = vec![1f64; 8];
        dscal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_dscal_int_large() {
        let mut x = vec![1f64; 100_000];
        let alpha = 2f64;
        let n = 100_000;
        let incx = 1;
        let expected_x = vec![2f64; 100_000];
        dscal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    //
    // cscal
    //
    #[test]
    fn test_cscal_inc_1_neg_alpha() {
        const C32_1_1: Complex32 = Complex32::new(1.0, 1.0);
        const C32_2_2: Complex32 = Complex32::new(2.0, -2.0);
        // C32_4_0 = C32_1_1 * C32_2_2
        const C32_4_0: Complex32 = Complex32::new(4.0, 0.0);

        let mut x = vec![C32_1_1; 8];
        let alpha = C32_2_2;
        let n = 8;
        let incx = 1;
        let expected_x = vec![C32_4_0; 8];
        cscal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_cscal_inc_n_offset() {
        const C32_1_1: Complex32 = Complex32::new(1.0, 1.0);
        const C32_2_2: Complex32 = Complex32::new(2.0, 2.0);
        // C32_0_4 = C32_1_1 * C32_2_2
        const C32_0_4: Complex32 = Complex32::new(0.0, 4.0);

        let mut x = vec![C32_1_1; 8];
        let alpha = C32_2_2;
        let n = 4;
        let incx = 2;
        let offset = 1;
        let expected_x = vec![C32_1_1, C32_0_4, C32_1_1, C32_0_4,
                              C32_1_1, C32_0_4, C32_1_1, C32_0_4];
        cscal(n, alpha, &mut x[offset..], incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_cscal_n_0() {
        const C32_1_1: Complex32 = Complex32::new(1.0, 1.0);
        const C32_2_2: Complex32 = Complex32::new(2.0, 2.0);

        let mut x = vec![C32_1_1; 8];
        let alpha = C32_2_2;
        let n = 0;
        let incx = 1;
        let expected_x = vec![C32_1_1; 8];
        cscal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_cscal_large() {
        const C32_1_1: Complex32 = Complex32::new(1.0, 1.0);
        const C32_2_2: Complex32 = Complex32::new(2.0, 2.0);
        // C32_0_4 = C32_1_1 * C32_2_2
        const C32_0_4: Complex32 = Complex32::new(0.0, 4.0);

        let mut x = vec![C32_1_1; 100_000];
        let alpha = C32_2_2;
        let n = 100_000;
        let incx = 1;
        let expected_x = vec![C32_0_4; 100_000];
        cscal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    //
    // zscal
    //
    #[test]
    fn test_zscal_inc_1_neg_alpha() {
        const C64_1_1: Complex64 = Complex64::new(1.0, 1.0);
        const C64_2_2: Complex64 = Complex64::new(2.0, -2.0);
        // C64_4_0 = C64_1_1 * C64_2_2
        const C64_4_0: Complex64 = Complex64::new(4.0, 0.0);

        let mut x = vec![C64_1_1; 8];
        let alpha = C64_2_2;
        let n = 8;
        let incx = 1;
        let expected_x = vec![C64_4_0; 8];
        zscal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_zscal_inc_n_offset() {
        const C64_1_1: Complex64 = Complex64::new(1.0, 1.0);
        const C64_2_2: Complex64 = Complex64::new(2.0, 2.0);
        // C64_0_4 = C64_1_1 * C64_2_2
        const C64_0_4: Complex64 = Complex64::new(0.0, 4.0);

        let mut x = vec![C64_1_1; 8];
        let alpha = C64_2_2;
        let n = 4;
        let incx = 2;
        let offset = 1;
        let expected_x = vec![C64_1_1, C64_0_4, C64_1_1, C64_0_4,
                              C64_1_1, C64_0_4, C64_1_1, C64_0_4];
        zscal(n, alpha, &mut x[offset..], incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_zscal_n_0() {
        const C64_1_1: Complex64 = Complex64::new(1.0, 1.0);
        const C64_2_2: Complex64 = Complex64::new(2.0, 2.0);

        let mut x = vec![C64_1_1; 8];
        let alpha = C64_2_2;
        let n = 0;
        let incx = 1;
        let expected_x = vec![C64_1_1; 8];
        zscal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }

    #[test]
    fn test_zscal_large() {
        const C64_1_1: Complex64 = Complex64::new(1.0, 1.0);
        const C64_2_2: Complex64 = Complex64::new(2.0, 2.0);
        // C64_0_4 = C64_1_1 * C64_2_2
        const C64_0_4: Complex64 = Complex64::new(0.0, 4.0);

        let mut x = vec![C64_1_1; 100_000];
        let alpha = C64_2_2;
        let n = 100_000;
        let incx = 1;
        let expected_x = vec![C64_0_4; 100_000];
        zscal(n, alpha, &mut x, incx);
        assert_eq!(x, expected_x);
    }
}
