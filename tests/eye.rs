use candle_ext::{
    candle::{DType, Device, Result, Tensor},
    TensorExt, F,
};

#[test]
fn test_eye() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::eye((3, 3), DType::U8, &device)?;
    #[rustfmt::skip]
    assert_eq!(a.to_vec2::<u8>()?, &[
        [1, 0, 0], 
        [0, 1, 0], 
        [0, 0, 1],
    ]);
    Ok(())
}

#[test]
fn test_eye_1() -> Result<()> {
    let device = Device::Cpu;
    let a = F::eye((3, 4), DType::F32, &device)?;
    #[rustfmt::skip]
    assert_eq!(a.to_vec2::<f32>()?, &[
        [1., 0., 0., 0.], 
        [0., 1., 0., 0.], 
        [0., 0., 1., 0.],
    ]);
    Ok(())
}

#[test]
fn test_eye_err() -> Result<()> {
    let device = Device::Cpu;
    let a = F::eye((3, 4, 5), DType::F32, &device);
    assert!(a.is_err());
    Ok(())
}
