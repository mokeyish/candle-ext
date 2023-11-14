use candle_ext::{
    candle::{Device, Result, Tensor},
    TensorExt,
};

#[test]
fn test_chunk2() -> Result<()> {
    let device = Device::Cpu;

    #[rustfmt::skip]
    let a = Tensor::new(&[
        [1u32,   2,  3,  4,  5,  6],
        [11u32, 22, 33, 14, 55, 66],
    ], &device)?;

    let (b, _) = a.chunk2(1)?;

    #[rustfmt::skip]
    assert_eq!(b.to_vec2::<u32>()?, &[
        [1,   2,  3], 
        [11, 22, 33]
    ]);

    Ok(())
}
