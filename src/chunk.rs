use crate::{
    candle::{shape::Dim, Result, Tensor},
    TensorVecExt, F,
};

impl F {
    #[inline]
    pub fn chunk2<D: Dim>(input: &Tensor, dim: D) -> Result<(Tensor, Tensor)> {
        input.chunk(2, dim)?.to_tuple2()
    }

    #[inline]
    pub fn chunk3<D: Dim>(input: &Tensor, dim: D) -> Result<(Tensor, Tensor, Tensor)> {
        input.chunk(3, dim)?.to_tuple3()
    }

    #[inline]
    pub fn chunk4<D: Dim>(input: &Tensor, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        input.chunk(4, dim)?.to_tuple4()
    }

    #[inline]
    pub fn chunk5<D: Dim>(
        input: &Tensor,
        dim: D,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        input.chunk(5, dim)?.to_tuple5()
    }
}
