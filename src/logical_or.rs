use crate::{
    candle::{
        cpu_backend::{binary_map, Map2U8},
        CustomOp2, Result, Tensor,
    },
    F,
};

impl F {
    /// Computes the element-wise logical OR of the given input tensors.
    /// Zeros are treated as False and nonzeros are treated as True.
    #[inline]
    pub fn logical_or(input: &Tensor, other: &Tensor) -> Result<Tensor> {
        input.apply_op2(other, LogicalOr)
    }
}

fn fwd<T: num_traits::Num, T2: num_traits::Num>(a: T, b: T2) -> u8 {
    if a.is_zero() && b.is_zero() {
        0
    } else {
        1
    }
}

struct LogicalOr;

const NAME: &str = "logical_or";

impl Map2U8 for LogicalOr {
    const OP: &'static str = NAME;

    fn f<T: candle_core::WithDType>(
        &self,
        v1: &[T],
        l1: &candle_core::Layout,
        v2: &[T],
        l2: &candle_core::Layout,
    ) -> Result<Vec<u8>> {
        Ok(binary_map(l1, l2, v1, v2, fwd))
    }
}

impl CustomOp2 for LogicalOr {
    fn name(&self) -> &'static str {
        NAME
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CpuStorage,
        l2: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        Ok((self.map(s1, l1, s2, l2)?, l1.shape().clone()))
    }
}
