use candle_ext::{
    candle::{Device, Result, Tensor},
    F,
};

#[test]
fn test_values_like() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::rand(-1f32, 1f32, (4, 4), &device)?;

    #[rustfmt::skip]
    assert_eq!(F::values_like(&a, 3f32)?.to_vec2::<f32>()?, &[
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
    ]);

    Ok(())
}

#[test]
fn test_values_like_1() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::ones((2, 2), candle_core::DType::U8, &device)?;

    #[rustfmt::skip]
    assert_eq!(F::values_like(&a, 3u8)?.to_vec2::<u8>()?, &[
        [3, 3], 
        [3, 3],
    ]);

    Ok(())
}
