use candle_ext::{
    candle::{DType, Device, Result, Tensor},
    TensorExt,
};

#[test]
fn test_logical_or_1() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::new(&[[1., 2.], [0., 4.]], &device)?;

    let b = Tensor::new(&[[0., 0.], [2., 1.]], &device)?;

    let c = a.logical_or(&b)?;

    assert_eq!(c.dtype(), DType::U8);

    #[rustfmt::skip]
    assert_eq!(c.to_vec2::<u8>()?, &[
        [1, 1],
        [1, 1],
    ]);
    Ok(())
}

#[test]
fn test_logical_or_bf16() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::new(&[[1., 2.], [0., 4.]], &device)?.to_dtype(DType::BF16)?;

    let b = Tensor::new(&[[0., 0.], [2., 1.]], &device)?.to_dtype(DType::BF16)?;

    #[rustfmt::skip]
    assert_eq!(a.logical_or(&b)?.to_vec2::<u8>()?, &[
        [1, 1],
        [1, 1],
    ]);

    Ok(())
}
