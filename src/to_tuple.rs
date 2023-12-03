use crate::{
    candle::{Error, Result, Tensor},
    F,
};

impl F {
    pub fn to_tuple2(mut tensors: Vec<Tensor>) -> Result<(Tensor, Tensor)> {
        check_tensor_vec_len(&tensors, 2)?;
        let b = tensors.pop().unwrap();
        let a = tensors.pop().unwrap();
        Ok((a, b))
    }

    pub fn to_tuple3(mut tensors: Vec<Tensor>) -> Result<(Tensor, Tensor, Tensor)> {
        check_tensor_vec_len(&tensors, 3)?;
        let c = tensors.pop().unwrap();
        let b = tensors.pop().unwrap();
        let a = tensors.pop().unwrap();
        Ok((a, b, c))
    }

    pub fn to_tuple4(mut tensors: Vec<Tensor>) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        check_tensor_vec_len(&tensors, 4)?;
        let d = tensors.pop().unwrap();
        let c = tensors.pop().unwrap();
        let b = tensors.pop().unwrap();
        let a = tensors.pop().unwrap();
        Ok((a, b, c, d))
    }

    pub fn to_tuple5(mut tensors: Vec<Tensor>) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        check_tensor_vec_len(&tensors, 5)?;
        let e = tensors.pop().unwrap();
        let d = tensors.pop().unwrap();
        let c = tensors.pop().unwrap();
        let b = tensors.pop().unwrap();
        let a = tensors.pop().unwrap();
        Ok((a, b, c, d, e))
    }
}

fn check_tensor_vec_len(tensors: &Vec<Tensor>, expected_len: usize) -> Result<()> {
    if tensors.len() != expected_len {
        Err(Error::Msg(format!(
            "unexpected len of Vec<Tensor>, expected: {}, got: {})",
            expected_len,
            tensors.len()
        )))
    } else {
        Ok(())
    }
}
