use candle_ext::{
    candle::{DType, Device, Result, Tensor},
    TensorExt, F,
};

#[test]
fn test_equal() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::rand(-1., 1., (2, 2), &device)?;
    assert!(F::equal(&a, &a)?);
    Ok(())
}

#[test]
fn test_equal_1() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::ones((2, 2), DType::U8, &device)?;
    let b = Tensor::ones((2, 2), DType::U8, &device)?;
    assert!(a.equal(&b)?);
    Ok(())
}

#[test]
fn test_equal_diffrent_shape() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::rand(-1., 1., (2, 2), &device)?;
    let b = Tensor::rand(-1., 1., (4, 2), &device)?;
    assert!(!Tensor::equal(&a, &b)?);
    Ok(())
}

#[test]
fn test_equal_diffrent_dtype() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::rand(-1., 1., (2, 2), &device)?;
    let b = Tensor::ones((2, 2), DType::U8, &device)?;
    assert!(!F::equal(&a, &b)?);
    Ok(())
}
