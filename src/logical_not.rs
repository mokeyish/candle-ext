use crate::{
    candle::{Result, Tensor},
    F,
};

impl F {
    pub fn logical_not(xs: &Tensor) -> Result<Tensor> {
        (xs - xs.ones_like()?)? * -1f64
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        candle::{DType, Device, Result, Tensor},
        TensorExt,
    };

    #[test]
    fn test_logical_not() -> Result<()> {
        let device = Device::Cpu;
        let a = Tensor::ones((4, 4), DType::F32, &device)?.triu(-1)?;
        assert_eq!(
            a.logical_not()?.to_vec2::<f32>()?,
            &[
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [1., 0., 0., 0.],
                [1., 1., 0., 0.]
            ]
        );
        Ok(())
    }
}
