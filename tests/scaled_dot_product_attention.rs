use candle_ext::{
    candle::{safetensors, Device, Result},
    F,
};

#[test]
fn test_scaled_dot_product_attention() -> Result<()> {
    let device = Device::Cpu;
    let tensor_map = safetensors::load(
        "data/scaled_dot_product_attention-qkvmo.safetensors",
        &device,
    )?;

    let q = tensor_map.get("q").unwrap();
    let k = tensor_map.get("k").unwrap();
    let v = tensor_map.get("v").unwrap();
    let m = tensor_map.get("m").unwrap();
    let o = tensor_map.get("o").unwrap();

    let out = F::scaled_dot_product_attention(q, k, v, Some(m), None, None, None)?;

    let diff = (o - out)?.sum_all()?.to_scalar::<f32>()?;
    assert!(diff < 0.000001f32);
    Ok(())
}

#[test]
fn test_scaled_dot_product_attention_1() -> Result<()> {
    let device = Device::Cpu;
    let tensor_map = safetensors::load(
        "data/scaled_dot_product_attention-qkvmo.safetensors",
        &device,
    )?;

    let q = tensor_map.get("q").unwrap();
    let k = tensor_map.get("k").unwrap();
    let v = tensor_map.get("v").unwrap();
    // let m = tensor_map.get("m").unwrap();
    let o = tensor_map.get("o").unwrap();

    let out = F::scaled_dot_product_attention(q, k, v, None, None, Some(true), None)?;

    let diff = (o - out)?.sum_all()?.to_scalar::<f32>()?;
    assert!(diff < 0.000001f32);
    Ok(())
}
