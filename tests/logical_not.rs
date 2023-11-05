use candle_ext::{
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
