use candle_ext::{
    candle::{DType, Device, Result, Tensor},
    TensorExt,
};

#[test]
fn test_masked_fill() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::ones((2, 2), DType::F32, &device)?;
    let m = Tensor::new(&[[1u8, 0], [1, 0]], &device)?;

    let b = a.masked_fill(&m, 3.)?;
    assert_eq!(b.to_vec2::<f32>()?, &[[3., 1.], [3., 1.]]);

    Ok(())
}

#[test]
fn test_masked_fill_2() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::ones((2, 2), DType::F32, &device)?;
    let m = Tensor::new(&[[1u8, 0], [0, 1]], &device)?;

    let b = a.masked_fill(&m, 3.)?;
    assert_eq!(b.to_vec2::<f32>()?, &[[3., 1.], [1., 3.]]);

    Ok(())
}

#[test]
fn test_masked_fill_3() -> Result<()> {
    let device = Device::Cpu;
    let weights = Tensor::ones((3, 2), DType::F32, &device)?;
    let weights_bias = weights.zeros_like()?;
    let mask = weights_bias.ones_like()?.tril(0)?.to_dtype(DType::U8)?;

    let weights_bias = weights_bias.masked_fill(&mask.logical_not()?, f32::NEG_INFINITY)?;

    let weights = (weights + weights_bias)?;
    assert_eq!(
        weights.to_vec2::<f32>()?,
        &[[1f32, f32::NEG_INFINITY], [1., 1.], [1., 1.],]
    );

    Ok(())
}
