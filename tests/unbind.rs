use candle_ext::{
    candle::{Device, Result, Tensor},
    F,
};

#[test]
fn test_unbind_dim0() -> Result<()> {
    let device = Device::Cpu;

    #[rustfmt::skip]
    let a = Tensor::new(&[
        [1u32, 2, 3], 
        [4u32, 5, 6], 
        [7u32, 8, 9]
    ], &device)?;

    let (a, b, c) = F::unbind3(&a, 0)?;
    assert_eq!(a.to_vec1::<u32>()?, &[1u32, 2, 3]);
    assert_eq!(b.to_vec1::<u32>()?, &[4u32, 5, 6]);
    assert_eq!(c.to_vec1::<u32>()?, &[7u32, 8, 9]);
    Ok(())
}

#[test]
fn test_unbind_dim1() -> Result<()> {
    let device = Device::Cpu;

    #[rustfmt::skip]
    let a = Tensor::new(&[
        [1u32, 2, 3], 
        [4u32, 5, 6], 
        [7u32, 8, 9]
    ], &device)?;

    let (a, b, c) = F::unbind3(&a, 1)?;
    assert_eq!(a.to_vec1::<u32>()?, &[1u32, 4, 7]);
    assert_eq!(b.to_vec1::<u32>()?, &[2u32, 5, 8]);
    assert_eq!(c.to_vec1::<u32>()?, &[3u32, 6, 9]);
    Ok(())
}

#[test]
fn test_unbind_error() -> Result<()> {
    let device = Device::Cpu;

    #[rustfmt::skip]
    let a = Tensor::new(&[
        [1u32, 2, 3], 
        [4u32, 5, 6], 
        [7u32, 8, 9],
    ], &device)?;

    assert!(F::unbind2(&a, 0).is_err());
    Ok(())
}
