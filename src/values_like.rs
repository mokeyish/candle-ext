use crate::{
    candle::{Result, Tensor, WithDType},
    F,
};

impl F {
    #[inline]
    pub fn values_like<D: WithDType>(xs: &Tensor, value: D) -> Result<Tensor> {
        Tensor::from_slice(&[value], 1, xs.device())?
            .to_dtype(xs.dtype())?
            .broadcast_as(xs.shape())
    }
}
