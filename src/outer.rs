use crate::{
    candle::{Result, Tensor, D},
    F,
};

impl F {
    /// Outer product of input and vec2.
    /// If input is a vector of size nn and vec2 is a vector of size mm,
    /// then out must be a matrix of size (n×m)(n×m).
    pub fn outer(input: &Tensor, vec2: &Tensor) -> Result<Tensor> {
        assert!(input.rank() == 1, "outer: input must be 1D tensor");
        assert!(vec2.rank() == 1, "outer: vec2 must be 1D tensor");
        let a_len = input.dim(D::Minus1)?;
        let b_len = vec2.dim(D::Minus1)?;

        let s = (a_len, b_len);

        let a = input.reshape((a_len, 1))?.broadcast_as(s)?;
        let b = vec2.reshape((1, b_len))?.broadcast_as(s)?;
        a * b
    }
}
