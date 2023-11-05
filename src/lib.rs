pub mod candle {
    pub use candle_core::*;
    pub use candle_nn as nn;
}

use candle::{Result, Tensor, WithDType};

mod logical_not;
mod masked_fill;
mod scaled_dot_product_attention;
mod triangular;
mod values_like;

pub struct F;

pub trait TensorExt: Sized {
    fn tril(&self, diagonal: isize) -> Result<Self>;
    fn triu(&self, diagonal: isize) -> Result<Self>;
    fn values_like<D: WithDType>(&self, value: D) -> Result<Self>;
    fn logical_not(&self) -> Result<Self>;
    fn masked_fill<D: WithDType>(&self, mask: &Tensor, value: D) -> Result<Self>;
}

impl TensorExt for Tensor {
    #[inline]
    fn tril(&self, diagonal: isize) -> Result<Self> {
        F::tril(self, diagonal)
    }

    #[inline]
    fn triu(&self, diagonal: isize) -> Result<Self> {
        F::triu(self, diagonal)
    }

    #[inline]
    fn logical_not(&self) -> Result<Self> {
        F::logical_not(self)
    }

    #[inline]
    fn values_like<D: WithDType>(&self, value: D) -> Result<Self> {
        F::values_like(self, value)
    }

    #[inline]
    fn masked_fill<D: WithDType>(&self, mask: &Tensor, value: D) -> Result<Self> {
        F::masked_fill(self, mask, value)
    }
}
