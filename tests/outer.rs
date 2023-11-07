use candle_ext::{
    candle::{Device, Result, Tensor},
    TensorExt,
};

#[test]
fn test_outer() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::arange(1f32, 4f32, &device)?;
    let b = Tensor::arange_step(3f32, 9f32, 2f32, &device)?;

    let c = a.outer(&b)?;
    #[rustfmt::skip]
    assert_eq!(c.to_vec2::<f32>()?, &[
        [3., 5., 7.], 
        [6., 10., 14.], 
        [9., 15., 21.],
    ]);
    Ok(())
}
