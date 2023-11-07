use crate::{
    candle::{Result, Tensor},
    F,
};

impl F {
    /// `True`` if two tensors have the same size and elements, False otherwise.
    pub fn equal(input: &Tensor, other: &Tensor) -> Result<bool> {
        Ok(
            if input.dtype() != other.dtype() || input.shape() != other.shape() {
                false
            } else {
                (input
                    .eq(other)?
                    .to_dtype(candle_core::DType::U32)?
                    .sum_all()?
                    .to_scalar::<u32>()?) as usize
                    == input.elem_count()
            },
        )
    }
}
