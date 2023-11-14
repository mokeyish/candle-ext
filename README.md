# Candle Extensions

![Test](https://github.com/mokeyish/candle_ext/actions/workflows/test.yml/badge.svg?branch=main)
[![](https://img.shields.io/crates/v/candle-ext.svg)](https://crates.io/crates/candle-ext)

An extension library to [Candle](https://github.com/huggingface/candle) that provides PyTorch functions not currently available in Candle

```rust
 use candle_ext::{
     candle::{ D, DType, Device, Result, Tensor},
     TensorExt, F,
 };

 fn main() -> Result<()> {
     let device = Device::Cpu;
     let q = Tensor::randn(0., 1., (3, 3, 2, 4), &device)?;
     let k = Tensor::randn(0., 1., (1, 3, 3, 4), &device)?;
     let v = Tensor::randn(0., 1., (1, 3, 3, 4), &device)?;
     let m = Tensor::ones((q.dim(D::Minus2)?, k.dim(D::Minus2)?), DType::U8, &device)?.tril(0)?;

     let o = F::scaled_dot_product_attention(&q, &k, &v, Some(&m), None, None, None)?;

     Ok(())
 }
```

Currently provides (see also [tests](https://github.com/mokeyish/candle-ext/tree/main/tests)):

- F::scaled_dot_product_attention

- F::chunk2..5 / Tensor::chunk2..5

- F::equal / Tensor::equal

- F::eye / Tensor::eye

- F::triu / Tensor::triu

- F::tril / Tensor::tril

- F::masked_fill / Tensor::masked_fill

- F::logical_not / Tensor::logical_not

- F::outer / Tensor::outer

- F::unbind / Tensor::unbind / F::unbind2..5 / Tensor::unbind2..5

- F::values_like / Tensor::values_like


## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
