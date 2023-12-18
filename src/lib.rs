#![allow(unused_imports)]
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
#[cfg(feature = "cuda")]
mod kernels;
mod logical_not;
mod logical_or;
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
    #[cfg(feature = "chunk")]
    fn chunk2<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor)>;
    #[cfg(feature = "chunk")]
    fn chunk3<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor)>;
    #[cfg(feature = "chunk")]
    fn chunk4<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor)>;
    #[cfg(feature = "chunk")]
    fn chunk5<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
    #[cfg(feature = "cumsum")]
    fn cumsum<D: Dim>(&self, dim: D) -> Result<Tensor>;
    #[cfg(feature = "equal")]
    fn equal(&self, other: &Tensor) -> Result<bool>;
    #[cfg(feature = "eye")]
    fn eye<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Tensor>;
    #[cfg(feature = "full")]
    fn full<S: Into<Shape>, D: WithDType>(
        shape: S,
        fill_value: D,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor>;
    #[cfg(feature = "full_like")]
    fn full_like<D: WithDType>(&self, fill_value: D) -> Result<Tensor>;
    #[cfg(feature = "logical_not")]
    fn logical_not(&self) -> Result<Self>;
    #[cfg(feature = "logical_or")]
    fn logical_or(&self, other: &Tensor) -> Result<Self>;
    #[cfg(feature = "masked_fill")]
    fn masked_fill<D: WithDType>(&self, mask: &Tensor, value: D) -> Result<Self>;
    #[cfg(feature = "outer")]
    fn outer(&self, vec2: &Tensor) -> Result<Self>;
    #[cfg(feature = "triangular")]
    fn tril(&self, diagonal: isize) -> Result<Self>;
    #[cfg(feature = "triangular")]
    fn triu(&self, diagonal: isize) -> Result<Self>;
    #[cfg(feature = "unbind")]
    fn unbind<D: Dim>(&self, dim: D) -> Result<Vec<Tensor>>;
    #[cfg(feature = "unbind")]
    fn unbind2<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor)>;
    #[cfg(feature = "unbind")]
    fn unbind3<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor)>;
    #[cfg(feature = "unbind")]
    fn unbind4<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor)>;
    #[cfg(feature = "unbind")]
    fn unbind5<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
}

impl TensorExt for Tensor {
    #[cfg(feature = "triangular")]
    #[inline]
    fn tril(&self, diagonal: isize) -> Result<Self> {
        F::tril(self, diagonal)
    }

    #[cfg(feature = "triangular")]
    #[inline]
    fn triu(&self, diagonal: isize) -> Result<Self> {
        F::triu(self, diagonal)
    }
    #[cfg(feature = "logical_not")]
    #[inline]
    fn logical_not(&self) -> Result<Self> {
        F::logical_not(self)
    }

    #[cfg(feature = "logical_or")]
    #[inline]
    fn logical_or(&self, other: &Tensor) -> Result<Self> {
        F::logical_or(self, other)
    }

    #[cfg(feature = "masked_fill")]
    #[inline]
    fn masked_fill<D: WithDType>(&self, mask: &Tensor, value: D) -> Result<Self> {
        F::masked_fill(self, mask, value)
    }

    #[cfg(feature = "outer")]
    #[inline]
    fn outer(&self, vec2: &Tensor) -> Result<Self> {
        F::outer(self, vec2)
    }

    #[cfg(feature = "unbind")]
    #[inline]
    fn unbind<D: Dim>(&self, dim: D) -> Result<Vec<Tensor>> {
        F::unbind(self, dim)
    }

    #[cfg(feature = "unbind")]
    #[inline]
    fn unbind2<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor)> {
        F::unbind2(self, dim)
    }

    #[cfg(feature = "unbind")]
    #[inline]
    fn unbind3<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor)> {
        F::unbind3(self, dim)
    }

    #[cfg(feature = "unbind")]
    #[inline]
    fn unbind4<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        F::unbind4(self, dim)
    }

    #[cfg(feature = "unbind")]
    #[inline]
    fn unbind5<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        F::unbind5(self, dim)
    }

    #[cfg(feature = "equal")]
    #[inline]
    fn equal(&self, other: &Tensor) -> Result<bool> {
        F::equal(self, other)
    }

    #[cfg(feature = "eye")]
    #[inline]
    fn eye<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Tensor> {
        F::eye(shape, dtype, device)
    }

    #[cfg(feature = "chunk")]
    #[inline]
    fn chunk2<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor)> {
        F::chunk2(self, dim)
    }

    #[cfg(feature = "chunk")]
    #[inline]
    fn chunk3<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor)> {
        F::chunk3(self, dim)
    }

    #[cfg(feature = "chunk")]
    #[inline]
    fn chunk4<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        F::chunk4(self, dim)
    }

    #[cfg(feature = "chunk")]
    #[inline]
    fn chunk5<D: Dim>(&self, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        F::chunk5(self, dim)
    }

    #[cfg(feature = "cumsum")]
    #[inline]
    fn cumsum<D: Dim>(&self, dim: D) -> Result<Tensor> {
        F::cumsum(self, dim)
    }

    #[cfg(feature = "full")]
    #[inline]
    fn full<S: Into<Shape>, D: WithDType>(
        shape: S,
        fill_value: D,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        F::full(shape, fill_value, dtype, device)
    }

    #[cfg(feature = "full_like")]
    #[inline]
    fn full_like<D: WithDType>(&self, fill_value: D) -> Result<Tensor> {
        F::full_like(self, fill_value)
    }
}

pub trait TensorVecExt {
    #[cfg(feature = "to_tuple")]
    fn to_tuple2(self) -> Result<(Tensor, Tensor)>;
    #[cfg(feature = "to_tuple")]
    fn to_tuple3(self) -> Result<(Tensor, Tensor, Tensor)>;
    #[cfg(feature = "to_tuple")]
    fn to_tuple4(self) -> Result<(Tensor, Tensor, Tensor, Tensor)>;
    #[cfg(feature = "to_tuple")]
    fn to_tuple5(self) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)>;
}

impl TensorVecExt for Vec<Tensor> {
    #[cfg(feature = "to_tuple")]
    #[inline]
    fn to_tuple2(self) -> Result<(Tensor, Tensor)> {
        F::to_tuple2(self)
    }

    #[cfg(feature = "to_tuple")]
    #[inline]
    fn to_tuple3(self) -> Result<(Tensor, Tensor, Tensor)> {
        F::to_tuple3(self)
    }

    #[cfg(feature = "to_tuple")]
    #[inline]
    fn to_tuple4(self) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        F::to_tuple4(self)
    }

    #[cfg(feature = "to_tuple")]
    #[inline]
    fn to_tuple5(self) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        F::to_tuple5(self)
    }
}
