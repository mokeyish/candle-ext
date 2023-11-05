use crate::{
    candle::{Result, Tensor},
    F,
};

impl F {
    #[inline]
    pub fn logical_not(xs: &Tensor) -> Result<Tensor> {
        (xs - xs.ones_like()?)? * -1f64
    }
}
