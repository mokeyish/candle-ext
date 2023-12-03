use crate::{
    candle::{DType, Device, Result, Shape, Tensor, WithDType},
    F,
};

impl F {
    /// Creates a tensor of size size filled with fill_value. The tensor’s dtype is inferred from fill_value.
    ///
    /// [https://pytorch.org/docs/stable/generated/torch.full.html](https://pytorch.org/docs/stable/generated/torch.full.html)
    pub fn full<S: Into<Shape>, D: WithDType>(
        shape: S,
        fill_value: D,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        Tensor::new(&[fill_value], device)?
            .to_dtype(dtype)?
            .broadcast_as(shape)
    }

    /// Returns a tensor with the same size as input filled with fill_value.
    ///
    /// F::full_like(input, fill_value) is equivalent to
    /// F::full(input.shape(), fill_value, dtype=input.dtype(), device=input.device()).
    ///
    /// [https://pytorch.org/docs/stable/generated/torch.full_like.html](https://pytorch.org/docs/stable/generated/torch.full_like.html)
    #[inline]
    pub fn full_like<D: WithDType>(input: &Tensor, fill_value: D) -> Result<Tensor> {
        F::full(input.shape(), fill_value, input.dtype(), input.device())
    }
}
