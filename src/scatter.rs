#![cfg(feature = "scatter")]

use crate::{
    candle::{
        self, backend::BackendStorage, shape::Dim, CustomOp3, Error, IntDType, Layout, Result,
        Tensor,
    },
    F,
};

impl F {
    /// Writes all values from the tensor src into self at the indices specified in the
    /// index tensor. For each value in src, its output index is specified by its index
    /// in src for dimension != dim and by the corresponding value in index for
    /// dimension = dim.
    pub fn scatter<D: Dim>(xs: &Tensor, indexes: &Tensor, src: &Tensor, dim: D) -> Result<Tensor> {
        xs.apply_op3(
            indexes,
            src,
            Scatter {
                dim: dim.to_index(xs.shape(), "scatter")?,
                op: ScatterOp::Assign,
            },
        )
    }

    pub fn scatter_mul<D: Dim>(
        xs: &Tensor,
        indexes: &Tensor,
        src: &Tensor,
        dim: D,
    ) -> Result<Tensor> {
        xs.apply_op3(
            indexes,
            src,
            Scatter {
                dim: dim.to_index(xs.shape(), "scatter")?,
                op: ScatterOp::Mul,
            },
        )
    }
}

const NAME: &str = "scatter";

struct Scatter {
    dim: usize,
    op: ScatterOp,
}

#[derive(Clone, Copy)]
enum ScatterOp {
    Assign,
    Mul,
}

impl ScatterOp {
    fn name(&self) -> &'static str {
        match self {
            ScatterOp::Assign => "scatter",
            ScatterOp::Mul => "scatter_mul",
        }
    }
}

impl CustomOp3 for Scatter {
    fn name(&self) -> &'static str {
        self.op.name()
    }

    fn cpu_fwd(
        &self,
        s: &candle::CpuStorage,
        l: &Layout,
        ids: &candle::CpuStorage,
        ids_l: &Layout,
        src: &candle::CpuStorage,
        src_l: &Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        use candle::{cpu_backend::Map2, CpuStorage, WithDType};

        struct ScatterOpParams<'a, I: IntDType> {
            ids: &'a [I],
            ids_l: &'a Layout,
            dim: usize,
            op: ScatterOp,
        }

        impl<'a, I: IntDType> Map2 for ScatterOpParams<'a, I> {
            const OP: &'static str = NAME;

            fn f<T: WithDType>(
                &self,
                v1: &[T],
                l1: &Layout,
                src: &[T],
                src_l: &Layout,
            ) -> Result<Vec<T>> {
                let dst_len = l1.shape().elem_count();
                let mut dst = vec![T::zero(); dst_len];

                {
                    let mut dst_s = T::to_cpu_storage_owned(dst);
                    T::to_cpu_storage(v1).copy_strided_src(&mut dst_s, 0, l1)?;
                    dst = T::cpu_storage_data(dst_s)?;
                }

                let src = match src_l.contiguous_offsets() {
                    None => Err(Error::RequiresContiguous { op: self.op.name() }.bt())?,
                    Some((o1, o2)) => &src[o1..o2],
                };

                let dim = self.dim;
                let ids_dims = self.ids_l.dims();
                let dst_dims = l1.dims();
                let dst_dim_len = dst_dims[dim];
                let dst_right_len: usize = dst_dims[dim + 1..].iter().product();

                let ids_left_len: usize = ids_dims[..dim].iter().product();
                let ids_dim_len = ids_dims[dim];
                let ids_right_len: usize = ids_dims[dim + 1..].iter().product();

                let ids = match self.ids_l.contiguous_offsets() {
                    Some((a, b)) => &self.ids[a..b],
                    None => Err(Error::RequiresContiguous { op: self.op.name() }.bt())?,
                };
                for left_i in 0..ids_left_len {
                    let start_ids_idx = left_i * ids_right_len * ids_dim_len;
                    let start_dst_idx = left_i * dst_right_len * dst_dim_len;
                    for i in 0..ids_dim_len {
                        let start_ids_idx = start_ids_idx + i * ids_right_len;
                        for right_i in 0..dst_right_len {
                            let ids_idx = start_ids_idx + right_i;
                            let index = ids[ids_idx].as_usize();
                            if index >= dst_dim_len {
                                Err(Error::InvalidIndex {
                                    index,
                                    size: dst_dim_len,
                                    op: self.op.name(),
                                }
                                .bt())?
                            }
                            let dst_idx = start_dst_idx + index * dst_right_len + right_i;
                            match self.op {
                                ScatterOp::Assign => dst[dst_idx] = src[ids_idx],
                                ScatterOp::Mul => dst[dst_idx] *= src[ids_idx],
                            }
                        }
                    }
                }

                Ok(dst)
            }
        }

        let storage = match ids {
            CpuStorage::U8(ids) => ScatterOpParams {
                ids,
                ids_l,
                dim: self.dim,
                op: self.op,
            }
            .map(s, l, src, src_l),
            CpuStorage::U32(ids) => ScatterOpParams {
                ids,
                ids_l,
                dim: self.dim,
                op: self.op,
            }
            .map(s, l, src, src_l),
            CpuStorage::I64(ids) => ScatterOpParams {
                ids,
                ids_l,
                dim: self.dim,
                op: self.op,
            }
            .map(s, l, src, src_l),
            _ => Err(Error::UnsupportedDTypeForOp(s.dtype(), "scatter-add")),
        }?;

        Ok((storage, l.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle::CudaStorage,
        l: &Layout,
        ids: &candle::CudaStorage,
        ids_l: &Layout,
        src: &candle::CudaStorage,
        src_l: &Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use crate::kernels;
        use candle::{
            backend::BackendDevice,
            cuda_backend::{
                cudarc::driver::{DevicePtr as _, LaunchAsync, LaunchConfig},
                kernel_name, CudaError, CudaStorage, CudaStorageSlice, Map2InPlace, WrapErr,
            },
            DType,
        };

        struct ScatterOpParams<'a> {
            ids: &'a CudaStorage,
            ids_l: &'a Layout,
            dim: usize,
            op: ScatterOp,
        }

        impl ScatterOp {}

        impl<'a> Map2InPlace for ScatterOpParams<'a> {
            fn f<
                T: candle_core::cuda_backend::cudarc::driver::DeviceRepr
                    + candle_core::WithDType
                    + candle_core::cuda_backend::cudarc::driver::ValidAsZeroBits,
            >(
                &self,
                dst: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
                dst_shape: &candle_core::Shape,
                src: &candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
                src_l: &Layout,
                dev: &candle_core::CudaDevice,
            ) -> Result<()> {
                let ids = self.ids;
                let ids_l = self.ids_l;
                let dim = self.dim;
                let (ids_o1, ids_o2) = match ids_l.contiguous_offsets() {
                    Some(o12) => o12,
                    None => Err(Error::RequiresContiguous { op: self.op.name() }.bt())?,
                };
                let (name, ids) = match &ids.slice {
                    CudaStorageSlice::U32(slice) => (
                        format!("{}_u32", self.op.name()),
                        *slice.slice(ids_o1..ids_o2).device_ptr(),
                    ),
                    CudaStorageSlice::I64(slice) => (
                        format!("{}_i64", self.op.name()),
                        *slice.slice(ids_o1..ids_o2).device_ptr(),
                    ),
                    CudaStorageSlice::U8(slice) => (
                        format!("{}_u8", self.op.name()),
                        *slice.slice(ids_o1..ids_o2).device_ptr(),
                    ),
                    _ => Err(CudaError::UnexpectedDType {
                        msg: "scatter ids should be u8/u32/i64",
                        expected: DType::U32,
                        got: ids.dtype(),
                    })?,
                };
                let src = match src_l.contiguous_offsets() {
                    Some((o1, o2)) => src.slice(o1..o2),
                    None => Err(Error::RequiresContiguous { op: self.op.name() }.bt())?,
                };
                let left_sz: usize = src_l.dims()[..dim].iter().product();
                let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
                let src_dim_sz = src_l.dims()[dim];
                let dst_dim_sz = dst_shape.dims()[dim];
                let cfg = LaunchConfig::for_num_elems((left_sz * right_sz) as u32);
                let func = dev.get_or_load_func(&kernel_name::<T>(&name), kernels::INDEXING)?;
                // SAFETY: Set later by running the kernel.
                let params = (ids, &src, dst, left_sz, src_dim_sz, dst_dim_sz, right_sz);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }.w()?;
                Ok(())
            }
        }

        let device = s.device().clone();
        let mut acc = device.zeros_impl(l.shape(), s.dtype())?;
        s.copy_strided_src(&mut acc, 0, l)?;

        ScatterOpParams {
            ids,
            ids_l,
            dim: self.dim,
            op: self.op,
        }
        .map(&mut acc.slice, l.shape(), &src.slice, src_l, &device)?;

        Ok((acc, l.shape().clone()))
    }
}
