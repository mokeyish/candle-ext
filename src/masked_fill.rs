use crate::{
    candle::{Result, Tensor, WithDType},
    TensorExt, F,
};

impl F {
    /// Fills elements of xs tensor with value where mask is True. The shape of mask must be broadcastable with the shape of the underlying tensor.
    pub fn masked_fill<D: WithDType>(xs: &Tensor, mask: &Tensor, value: D) -> Result<Tensor> {
        let dtype = xs.dtype();
        let shape = xs.shape();
        let on_true = xs.values_like(value)?.to_dtype(dtype)?;
        let on_false = xs;
        mask.broadcast_as(shape)?.where_cond(&on_true, on_false)
    }
}
