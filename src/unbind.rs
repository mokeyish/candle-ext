use crate::{
    candle::{shape::Dim, Error, Result, Tensor},
    F,
};

impl F {
    /// Removes a tensor dimension.
    ///
    /// Returns a tuple of all slices along a given dimension, already without it.
    pub fn unbind<D: Dim>(input: &Tensor, dim: D) -> Result<Vec<Tensor>> {
        let dim = dim.to_index(input.shape(), "unbind")?;
        let mut tensors = vec![];
        for i in 0..input.dim(dim)? {
            tensors.push(input.narrow(dim, i, 1)?.squeeze(dim)?);
        }
        Ok(tensors)
    }

    pub fn unbind2<D: Dim>(input: &Tensor, dim: D) -> Result<(Tensor, Tensor)> {
        let dim = dim.to_index(input.shape(), "unbind")?;
        check_unbind_shape(input, dim, 2)?;
        let tensors = F::unbind(input, dim)?;
        F::unbind_vec2(tensors)
    }

    pub fn unbind3<D: Dim>(input: &Tensor, dim: D) -> Result<(Tensor, Tensor, Tensor)> {
        let dim = dim.to_index(input.shape(), "unbind")?;
        check_unbind_shape(input, dim, 3)?;
        let mut tensors = F::unbind(input, dim)?;
        let c = tensors.pop().unwrap();
        let b = tensors.pop().unwrap();
        let a = tensors.pop().unwrap();
        Ok((a, b, c))
    }

    pub fn unbind4<D: Dim>(input: &Tensor, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let dim = dim.to_index(input.shape(), "unbind")?;
        check_unbind_shape(input, dim, 4)?;
        let tensors = F::unbind(input, dim)?;
        F::unbind_vec4(tensors)
    }
    pub fn unbind5<D: Dim>(
        input: &Tensor,
        dim: D,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let dim = dim.to_index(input.shape(), "unbind")?;
        check_unbind_shape(input, dim, 5)?;
        let tensors = F::unbind(input, dim)?;
        F::unbind_vec5(tensors)
    }

    pub fn unbind_vec2(mut tensors: Vec<Tensor>) -> Result<(Tensor, Tensor)> {
        check_tensor_vec_len(&tensors, 2)?;
        let b = tensors.pop().unwrap();
        let a = tensors.pop().unwrap();
        Ok((a, b))
    }

    pub fn unbind_vec3(mut tensors: Vec<Tensor>) -> Result<(Tensor, Tensor, Tensor)> {
        check_tensor_vec_len(&tensors, 3)?;
        let c = tensors.pop().unwrap();
        let b = tensors.pop().unwrap();
        let a = tensors.pop().unwrap();
        Ok((a, b, c))
    }

    pub fn unbind_vec4(mut tensors: Vec<Tensor>) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        check_tensor_vec_len(&tensors, 4)?;
        let d = tensors.pop().unwrap();
        let c = tensors.pop().unwrap();
        let b = tensors.pop().unwrap();
        let a = tensors.pop().unwrap();
        Ok((a, b, c, d))
    }

    pub fn unbind_vec5(
        mut tensors: Vec<Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
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

fn check_unbind_shape(input: &Tensor, dim: usize, expected_len: usize) -> Result<()> {
    if input.dim(dim)? != expected_len {
        let got_shape = input.shape().clone();
        let mut expected_shape = got_shape.dims().to_vec();
        let got_len = expected_shape[dim];
        expected_shape[dim] = expected_len;
        Err(Error::UnexpectedShape {
            msg: format!(
                "unbind{expected_len} failed, expected dim len {expected_len}, got {got_len}"
            ),
            expected: expected_shape.into(),
            got: got_shape,
        }
        .bt())
    } else {
        Ok(())
    }
}
