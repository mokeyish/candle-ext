use candle_ext::{
    candle::{Device, Result, Tensor},
    TensorExt, F,
};

#[test]
fn test_cumsum() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::new(&[1u32, 2, 3], &device)?;
    let b = F::cumsum(&a, 0)?;
    assert_eq!(b.to_vec1::<u32>()?, &[1, 3, 6]);
    Ok(())
}

#[test]
fn test_cumsum_dim_0() -> Result<()> {
    let device = Device::Cpu;
    #[rustfmt::skip]
    let a = Tensor::new(&[
        [1u32, 2, 3],
        [4u32, 5, 6],
        [7u32, 8, 9],
    ], &device)?;
    let b = a.cumsum(0)?;

    println!("{}", b);

    #[rustfmt::skip]
    assert_eq!(b.to_vec2::<u32>()?, &[
        [ 1,  2,  3],
        [ 5,  7,  9],
        [12, 15, 18]
    ]);
    Ok(())
}

#[test]
fn test_cumsum_dim_1() -> Result<()> {
    let device = Device::Cpu;
    #[rustfmt::skip]
    let a = Tensor::new(&[
        [1u32, 2, 3],
        [4u32, 5, 6],
        [7u32, 8, 9],
    ], &device)?;
    let b = a.cumsum(1)?;

    println!("{}", b);

    #[rustfmt::skip]
    assert_eq!(b.to_vec2::<u32>()?, &[
        [1,  3,  6],
        [4,  9, 15],
        [7, 15, 24]
    ]);
    Ok(())
}
