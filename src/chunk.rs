use crate::{
    candle::{shape::Dim, Result, Tensor},
    TensorVecExt, F,
};

impl F {
    #[inline]
    pub fn chunk2<D: Dim>(input: &Tensor, dim: D) -> Result<(Tensor, Tensor)> {
        input.chunk(2, dim)?.unbind2()
    }

    #[inline]
    pub fn chunk3<D: Dim>(input: &Tensor, dim: D) -> Result<(Tensor, Tensor, Tensor)> {
        input.chunk(3, dim)?.unbind3()
    }

    #[inline]
    pub fn chunk4<D: Dim>(input: &Tensor, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        input.chunk(4, dim)?.unbind4()
    }

    #[inline]
    pub fn chunk5<D: Dim>(
        input: &Tensor,
        dim: D,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        input.chunk(5, dim)?.unbind5()
    }
}
