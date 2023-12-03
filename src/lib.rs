//! An extension library to [Candle](https://github.com/huggingface/candle) that provides PyTorch functions not currently available in Candle
//!
//! # Examples
//!
//! ```no_run
//! use candle_ext::{
//!     candle::{ D, DType, Device, Result, Tensor},
//!     TensorExt, F,
//! };
//!
//! fn main() -> Result<()> {
//!     let device = Device::Cpu;
//!     let q = Tensor::randn(0., 1., (3, 3, 2, 4), &device)?;
//!     let k = Tensor::randn(0., 1., (1, 3, 3, 4), &device)?;
//!     let v = Tensor::randn(0., 1., (1, 3, 3, 4), &device)?;
//!     let m = Tensor::ones((q.dim(D::Minus2)?, k.dim(D::Minus2)?), DType::U8, &device)?.tril(0)?;
//!
//!     let o = F::scaled_dot_product_attention(&q, &k, &v, Some(&m), None, None, None)?;
//!
//!     Ok(())
//! }
//! ```

pub mod candle {
    pub use candle_core::*;
    pub use candle_nn as nn;
}

use candle::{shape::Dim, DType, Device, Result, Shape, Tensor, WithDType};

mod chunk;
mod cumsum;
mod equal;
mod eye;
mod full;
mod logical_not;
mod masked_fill;
mod outer;
mod scaled_dot_product_attention;
mod to_tuple;
mod triangular;
mod unbind;

/// Tensor functional
/// # Examples
///
/// ```no_run
/// use candle_ext::{
///     candle::{ D, DType, Device, Result, Tensor},
///     TensorExt, F,
/// };
///
/// fn main() -> Result<()> {
///     let device = Device::Cpu;
///     let q = Tensor::randn(0., 1., (3, 3, 2, 4), &device)?;
///     let k = Tensor::randn(0., 1., (1, 3, 3, 4), &device)?;
///     let v = Tensor::randn(0., 1., (1, 3, 3, 4), &device)?;
///     let m = Tensor::ones((q.dim(D::Minus2)?, k.dim(D::Minus2)?), DType::U8, &device)?.tril(0)?;
///
///     let o = F::scaled_dot_product_attention(&q, &k, &v, Some(&m), None, None, None)?;
///
///     Ok(())
/// }
/// ```
pub struct F;

pub trait TensorExt: Sized {
    fn chunk2<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor)>;
    fn chunk3<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor)>;
    fn chunk4<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor)>;
    fn chunk5<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
    fn cumsum<D: Dim>(&self, dim: D) -> Result<Tensor>;
    fn equal(&self, other: &Tensor) -> Result<bool>;
    fn eye<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Tensor>;
    fn full<S: Into<Shape>, D: WithDType>(
        shape: S,
        fill_value: D,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor>;
    fn full_like<D: WithDType>(&self, fill_value: D) -> Result<Tensor>;
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

    #[inline]
    fn chunk2<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor)> {
        F::chunk2(self, dim)
    }

    #[inline]
    fn chunk3<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor)> {
        F::chunk3(self, dim)
    }

    #[inline]
    fn chunk4<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        F::chunk4(self, dim)
    }

    #[inline]
    fn chunk5<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        F::chunk5(self, dim)
    }

    #[inline]
    fn cumsum<D: Dim>(&self, dim: D) -> Result<Tensor> {
        F::cumsum(self, dim)
    }

    #[inline]
    fn full<S: Into<Shape>, D: WithDType>(
        shape: S,
        fill_value: D,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        F::full(shape, fill_value, dtype, device)
    }

    #[inline]
    fn full_like<D: WithDType>(&self, fill_value: D) -> Result<Tensor> {
        F::full_like(self, fill_value)
    }
}

pub trait TensorVecExt {
    fn to_tuple2(self) -> Result<(Tensor, Tensor)>;
    fn to_tuple3(self) -> Result<(Tensor, Tensor, Tensor)>;
    fn to_tuple4(self) -> Result<(Tensor, Tensor, Tensor, Tensor)>;
    fn to_tuple5(self) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
}

impl TensorVecExt for Vec<Tensor> {
    #[inline]
    fn to_tuple2(self) -> Result<(Tensor, Tensor)> {
        F::to_tuple2(self)
    }
    #[inline]
    fn to_tuple3(self) -> Result<(Tensor, Tensor, Tensor)> {
        F::to_tuple3(self)
    }
    #[inline]
    fn to_tuple4(self) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        F::to_tuple4(self)
    }
    #[inline]
    fn to_tuple5(self) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        F::to_tuple5(self)
    }
}
