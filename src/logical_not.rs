use crate::{
    candle::{Result, Tensor},
    F,
};

impl F {
    /// Computes the element-wise logical NOT of the given input tensor.
    /// If not specified, the output tensor will have the bool dtype.
    /// If the input tensor is not a bool tensor, zeros are treated as False and non-zeros are treated as True.
    #[inline]
    pub fn logical_not(xs: &Tensor) -> Result<Tensor> {
        xs.where_cond(&xs.zeros_like()?, &xs.ones_like()?)?
            .to_dtype(xs.dtype())
    }
}
