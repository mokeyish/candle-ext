pub mod candle {
    pub use candle_core::*;
    pub use candle_nn as nn;
}

use candle::{shape::Dim, DType, Device, Result, Shape, Tensor, WithDType};

mod equal;
mod eye;
mod logical_not;
mod masked_fill;
mod outer;
mod scaled_dot_product_attention;
mod triangular;
mod unbind;
mod values_like;

pub struct F;

pub trait TensorExt: Sized {
    fn equal(&self, other: &Tensor) -> Result<bool>;
    fn eye<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Tensor>;
    fn logical_not(&self) -> Result<Self>;
    fn masked_fill<D: WithDType>(&self, mask: &Tensor, value: D) -> Result<Self>;
    fn outer(&self, vec2: &Tensor) -> Result<Self>;
    fn tril(&self, diagonal: isize) -> Result<Self>;
    fn triu(&self, diagonal: isize) -> Result<Self>;
    fn unbind<D: Dim>(&self, dim: D) -> Result<Vec<Tensor>>;
    fn unbind2<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor)>;
    fn unbind3<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor)>;
    fn unbind4<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor)>;
    fn unbind5<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
    fn values_like<D: WithDType>(&self, value: D) -> Result<Self>;
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

    #[inline]
    fn outer(&self, vec2: &Tensor) -> Result<Self> {
        F::outer(self, vec2)
    }

    #[inline]
    fn unbind<D: Dim>(&self, dim: D) -> Result<Vec<Tensor>> {
        F::unbind(self, dim)
    }

    #[inline]
    fn unbind2<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor)> {
        F::unbind2(self, dim)
    }

    #[inline]
    fn unbind3<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor)> {
        F::unbind3(self, dim)
    }

    #[inline]
    fn unbind4<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        F::unbind4(self, dim)
    }

    #[inline]
    fn unbind5<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        F::unbind5(self, dim)
    }

    #[inline]
    fn equal(&self, other: &Tensor) -> Result<bool> {
        F::equal(self, other)
    }

    #[inline]
    fn eye<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Tensor> {
        F::eye(shape, dtype, device)
    }
}
