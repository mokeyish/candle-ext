use crate::{
    candle::{nn::ops, DType, Result, Tensor, D},
    TensorExt, F,
};

impl F {
    /// Computes scaled dot product attention on query, key and value tensors,
    /// using an optional attention mask if passed, and applying dropout
    /// if a probability greater than 0.0 is specified.
    ///
    /// # Arguments
    /// - query   - Query tensor; shape (N, ..., L, E)
    /// - key     - Key tensor; shape (N, ..., S, E)
    /// - value   - Value tensor; shape (N, ..., S, E)
    ///
    /// https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    /// # Errors
    ///
    /// This function will return an error if .
    pub fn scaled_dot_product_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        dropout_p: Option<f32>,
        is_causal: Option<bool>,
        scale: Option<f64>,
    ) -> Result<Tensor> {
        let device = query.device();
        let l = query.dim(D::Minus2)?;
        let s = key.dim(D::Minus2)?;
        let dim = query.dim(D::Minus1)?;

        let scale_factor = if let Some(scale) = scale {
            scale
        } else {
            1.0 / (dim as f64).sqrt()
        };

        let mut attn_bias = Tensor::zeros((l, s), query.dtype(), device)?;

        if matches!(is_causal, Some(true)) {
            assert!(attn_mask.is_none(), "scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
            let mask = Tensor::ones((l, s), DType::U8, device)?.tril(0)?;
            attn_bias = attn_bias.masked_fill(&mask.logical_not()?, f32::NEG_INFINITY)?;
        }

        if let Some(attn_mask) = attn_mask {
            if attn_mask.rank() > attn_bias.rank() {
                attn_bias = attn_bias.broadcast_as(attn_mask.shape())?;
            }
            if attn_mask.dtype() == DType::U8 {
                // bool
                attn_bias = attn_bias.masked_fill(&attn_mask.logical_not()?, f32::NEG_INFINITY)?;
            } else {
                attn_bias = (&attn_bias
                    + attn_mask
                        .to_dtype(attn_bias.dtype())?
                        .broadcast_as(attn_bias.shape())?)?;
            }
        }

        let mut attn_weights =
            (query.matmul(&key.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale_factor)?;

        attn_weights = (&attn_weights + attn_bias.broadcast_as(attn_weights.shape())?)?;
        attn_weights = ops::softmax_last_dim(&attn_weights)?;

        if let Some(drop_p) = dropout_p {
            attn_weights = ops::dropout(&attn_weights, drop_p)?;
        }
        attn_weights.matmul(value)
    }
}
