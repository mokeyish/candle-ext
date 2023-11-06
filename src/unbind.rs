use crate::{
    candle::{shape::Dim, Error, IndexOp, Result, Tensor},
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
            tensors.push(match dim {
                0 => input.i(i),
                1 => input.i((.., i)),
                2 => input.i((.., .., i)),
                3 => input.i((.., .., .., i)),
                4 => input.i((.., .., .., .., i)),
                5 => input.i((.., .., .., .., .., i)),
                6 => input.i((.., .., .., .., .., .., i)),
                _ => {
                    unimplemented!("unbind")
                }
            }?);
        }
        Ok(tensors)
    }

    pub fn unbind2<D: Dim>(input: &Tensor, dim: D) -> Result<(Tensor, Tensor)> {
        let dim = dim.to_index(input.shape(), "unbind")?;
        check_unbind_shape(input, dim, 2)?;
        let mut tensors = F::unbind(input, dim)?;
        let b = tensors.pop().unwrap();
        let a = tensors.pop().unwrap();
        Ok((a, b))
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
        let mut tensors = F::unbind(input, dim)?;
        let d = tensors.pop().unwrap();
        let c = tensors.pop().unwrap();
        let b = tensors.pop().unwrap();
        let a = tensors.pop().unwrap();
        Ok((a, b, c, d))
    }
    pub fn unbind5<D: Dim>(
        input: &Tensor,
        dim: D,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let dim = dim.to_index(input.shape(), "unbind")?;
        check_unbind_shape(input, dim, 5)?;
        let mut tensors = F::unbind(input, dim)?;
        let e = tensors.pop().unwrap();
        let d = tensors.pop().unwrap();
        let c = tensors.pop().unwrap();
        let b = tensors.pop().unwrap();
        let a = tensors.pop().unwrap();
        Ok((a, b, c, d, e))
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
