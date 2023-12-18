#![cfg(feature = "logical_not")]
use candle_ext::{
    candle::{Device, Result, Tensor},
    TensorExt,
};

#[test]
fn test_logical_not_1() -> Result<()> {
    let device = Device::Cpu;

    #[rustfmt::skip]
    let a = Tensor::new(&[
        [1u8, 1],
        [0,   1]
    ],&device)?;

    println!("{}", a.logical_not()?);

    #[rustfmt::skip]
    assert_eq!(a.logical_not()?.to_vec2::<u8>()?, &[
        [0, 0], 
        [1, 0],
    ]);
    Ok(())
}
