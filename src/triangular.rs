#![cfg(feature = "triangular")]
use crate::{
    candle::{Result, Tensor},
    F,
};

impl F {
    #[inline]
    pub fn triu(xs: &Tensor, diagonal: isize) -> Result<Tensor> {
        Self::apply_triangular(xs, diagonal, true)
    }

    /// Returns the lower triangular part of the matrix (2-D tensor) or
    /// batch of matrices input, the other elements of the result tensor
    /// out are set to 0.
    ///
    /// The lower triangular part of the matrix is defined as the
    /// elements on and below the diagonal.
    #[inline]
    pub fn tril(xs: &Tensor, diagonal: isize) -> Result<Tensor> {
        Self::apply_triangular(xs, diagonal, false)
    }

    fn apply_triangular(xs: &Tensor, diagonal: isize, upper: bool) -> Result<Tensor> {
        let device = xs.device();
        let (l, s) = xs.dims2()?;
        let mut xs_tri = vec![];
        for i in 0..l as isize {
            for j in 0..s as isize {
                let cond = if upper {
                    i + diagonal > j
                } else {
                    i + diagonal < j
                };
                xs_tri.push(if cond { 0u8 } else { 1u8 });
            }
        }
        xs * Tensor::from_vec(xs_tri, (l, s), device)?.to_dtype(xs.dtype())?
    }
}
