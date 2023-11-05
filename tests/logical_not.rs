use candle_ext::{
    candle::{DType, Device, Result, Tensor},
    TensorExt,
};

#[test]
fn test_logical_not_1() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::ones((2, 2), DType::U8, &device)?.triu(0)?;
    assert_eq!(a.logical_not()?.to_vec2::<u8>()?, &[[0, 0], [1, 0],]);
    Ok(())
}
