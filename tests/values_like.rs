use candle_ext::{
    candle::{Device, Result, Tensor},
    F,
};

#[test]
fn test_values_like() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::rand(-1., 1., (4, 4), &device)?;
    assert_eq!(
        F::values_like(&a, 3f32)?.to_vec2::<f32>()?,
        &[
            [3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.]
        ]
    );
    Ok(())
}
