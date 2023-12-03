use crate::{
    candle::{shape::Dim, Error, Result, Tensor},
    TensorVecExt, F,
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
        F::unbind(input, dim)?.to_tuple2()
    }

    pub fn unbind3<D: Dim>(input: &Tensor, dim: D) -> Result<(Tensor, Tensor, Tensor)> {
        let dim = dim.to_index(input.shape(), "unbind")?;
        check_unbind_shape(input, dim, 3)?;
        F::unbind(input, dim)?.to_tuple3()
    }

    pub fn unbind4<D: Dim>(input: &Tensor, dim: D) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let dim = dim.to_index(input.shape(), "unbind")?;
        check_unbind_shape(input, dim, 4)?;
        F::unbind(input, dim)?.to_tuple4()
    }
    pub fn unbind5<D: Dim>(
        input: &Tensor,
        dim: D,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let dim = dim.to_index(input.shape(), "unbind")?;
        check_unbind_shape(input, dim, 5)?;
        F::unbind(input, dim)?.to_tuple5()
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
