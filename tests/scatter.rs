#![cfg(feature = "scatter")]
use candle_ext::{
    candle::{test_device, DType, Device, Result, Tensor},
    TensorExt,
};

fn scatter(device: &Device) -> Result<()> {
    let t: Tensor = Tensor::arange(0f32, 12f32, device)?.reshape((4, 3))?;
    assert_eq!(
        t.to_vec2::<f32>()?,
        &[
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ]
    );
    let ids = Tensor::new(&[[0u32, 1, 2], [3, 4, 0], [3, 3, 1], [2, 0, 4]], device)?;
    let init = Tensor::ones((4, 5), DType::F32, device)?;
    let hs = init.scatter(&ids, &t, 1)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [0., 1., 2., 1., 1.],
            [5., 1., 1., 3., 4.],
            [1., 8., 1., 7., 1.],
            [10., 1., 9., 1., 11.]
        ]
    );

    let init = Tensor::ones((6, 3), DType::F32, device)?;
    let hs = init.scatter(&ids, &t, 0)?;
    assert_eq!(
        hs.to_vec2::<f32>()?,
        &[
            [0., 10., 5.],
            [1., 1., 8.],
            [9., 1., 2.],
            [6., 7., 1.],
            [1., 4., 11.],
            [1., 1., 1.]
        ]
    );
    Ok(())
}

test_device!(scatter, scatter_cpu, scatter_gpu, scatter_metal);
