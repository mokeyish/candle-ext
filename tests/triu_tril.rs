#![cfg(feature = "triangular")]
use candle_ext::{
    candle::{DType, Device, Result, Tensor},
    TensorExt, F,
};

#[test]
fn test_tril() -> Result<()> {
    let device = Device::Cpu;
    let a = F::tril(&Tensor::ones((4, 4), DType::F32, &device)?, 0)?;

    #[rustfmt::skip]
    assert_eq!(a.to_vec2::<f32>()?, &[
        [1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.],
    ]);

    Ok(())
}

#[test]
fn test_tril_diagonal_1() -> Result<()> {
    let device = Device::Cpu;
    let a = Tensor::ones((4, 4), DType::F32, &device)?.tril(1)?;

    #[rustfmt::skip]
    assert_eq!(a.to_vec2::<f32>()?, &[
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
    ]);
    Ok(())
}

#[test]
fn test_tril_diagonal_neg_1() -> Result<()> {
    let device = Device::Cpu;
    let a = F::tril(&Tensor::ones((4, 4), DType::F32, &device)?, -1)?;

    #[rustfmt::skip]
    assert_eq!(a.to_vec2::<f32>()?, &[
        [0., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
    ]);
    Ok(())
}

#[test]
fn test_triu() -> Result<()> {
    let device = Device::Cpu;
    let a = F::triu(&Tensor::ones((4, 4), DType::F32, &device)?, 0)?;

    #[rustfmt::skip]
    assert_eq!(a.to_vec2::<f32>()?, &[
        [1., 1., 1., 1.],
        [0., 1., 1., 1.],
        [0., 0., 1., 1.],
        [0., 0., 0., 1.],
    ]);

    Ok(())
}

#[test]
fn test_triu_diagonal_1() -> Result<()> {
    let device = Device::Cpu;
    let a = F::triu(&Tensor::ones((4, 4), DType::F32, &device)?, 1)?;

    #[rustfmt::skip]
    assert_eq!(a.to_vec2::<f32>()?, &[
        [0., 1., 1., 1.],
        [0., 0., 1., 1.],
        [0., 0., 0., 1.],
        [0., 0., 0., 0.],
    ]);
    Ok(())
}

#[test]
fn test_triu_diagonal_neg_1() -> Result<()> {
    let device = Device::Cpu;
    let a = F::triu(&Tensor::ones((4, 4), DType::F32, &device)?, -1)?;

    #[rustfmt::skip]
    assert_eq!(a.to_vec2::<f32>()?, &[
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [0., 1., 1., 1.],
        [0., 0., 1., 1.],
    ]);
    Ok(())
}
