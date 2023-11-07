use crate::{
    candle::{DType, Device, Result, Shape, Tensor},
    F,
};

impl F {
    /// Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
    pub fn eye<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Tensor> {
        let shape: Shape = shape.into();
        assert!(shape.rank() >= 1 && shape.rank() <= 2, "");
        let (n, m) = if shape.rank() == 1 {
            let dim = shape.dims1()?;
            (dim, dim)
        } else {
            shape.dims2()?
        };

        let mut xs = vec![];
        for i in 0..n as isize {
            for j in 0..m as isize {
                xs.push(if i == j { 1u8 } else { 0u8 });
            }
        }
        Tensor::from_vec(xs, (n, m), device)?.to_dtype(dtype)
    }
}
