use crate::{
    candle::{shape::Dim, Result, Tensor},
    F,
};

impl F {
    /// Returns the cumulative sum of elements of input in the dimension dim.
    ///
    /// [https://pytorch.org/docs/stable/generated/torch.cumsum.html](https://pytorch.org/docs/stable/generated/torch.cumsum.html)
    pub fn cumsum<D: Dim>(input: &Tensor, dim: D) -> Result<Tensor> {
        let dim = dim.to_index(input.shape(), "cumsum")?;
        let dim_size = input.dim(dim)?;

        let mut tensors = Vec::with_capacity(dim_size);

        let mut a = input.clone();
        for i in 0..dim_size {
            if i > 0 {
                a = a.narrow(dim, 1, dim_size - i)?;
                let b = input.narrow(dim, 0, dim_size - i)?;
                a = (a + b)?;
            }
            tensors.push(a.narrow(dim, 0, 1)?);
        }
        let cumsum = Tensor::cat(&tensors, dim)?;
        Ok(cumsum)
    }
}
