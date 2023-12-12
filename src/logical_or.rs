use crate::{
    candle::{self, CpuStorage, CustomOp2, Layout, Result, Shape, Tensor, WithDType},
    F,
};

impl F {
    /// Computes the element-wise logical OR of the given input tensors.
    /// Zeros are treated as False and nonzeros are treated as True.
    #[inline]
    pub fn logical_or(input: &Tensor, other: &Tensor) -> Result<Tensor> {
        input.apply_op2(other, LogicalOr)
    }
}

fn fwd<T: num_traits::Num, T2: num_traits::Num>(a: T, b: T2) -> u8 {
    if a.is_zero() && b.is_zero() {
        0
    } else {
        1
    }
}

struct LogicalOr;

const NAME: &str = "logical_or";

impl CustomOp2 for LogicalOr {
    fn name(&self) -> &'static str {
        NAME
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::cpu_backend::{binary_map, Map2U8};
        impl Map2U8 for LogicalOr {
            const OP: &'static str = NAME;

            fn f<T: WithDType>(
                &self,
                v1: &[T],
                l1: &Layout,
                v2: &[T],
                l2: &Layout,
            ) -> Result<Vec<u8>> {
                Ok(binary_map(l1, l2, v1, v2, fwd))
            }
        }
        Ok((self.map(s1, l1, s2, l2)?, l1.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use crate::candle::cuda_backend::{
            cudarc::driver::{CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits},
            CudaStorage, CudaStorageSlice, Map2Any, WrapErr,
        };
        use crate::candle::CudaDevice;
        use crate::kernels;

        impl Map2Any for LogicalOr {
            fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
                &self,
                src1: &CudaSlice<T>,
                layout1: &Layout,
                src2: &CudaSlice<T>,
                layout2: &Layout,
                device: &CudaDevice,
            ) -> Result<CudaStorageSlice> {
                let shape1 = layout1.shape();
                let dims1 = shape1.dims();
                let elem_count1 = shape1.elem_count();
                let launch_config = LaunchConfig::for_num_elems(elem_count1 as u32);
                let dims_and_strides = device
                    .htod_copy([dims1, layout1.stride(), layout2.stride()].concat())
                    .w()?;
                let src1 = src1.slice(layout1.start_offset()..);
                let src2 = src2.slice(layout2.start_offset()..);
                let dtype = T::DTYPE.as_str();
                let func =
                    device.get_or_load_func(&format!("{NAME}_{dtype}"), kernels::CUSTOM_BINARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { device.alloc::<u8>(elem_count1) }.w()?;
                let params = (
                    elem_count1,
                    dims1.len(),
                    &dims_and_strides,
                    &src1,
                    &src2,
                    &out,
                );
                // SAFETY: ffi
                unsafe { func.launch(launch_config, params) }.w()?;

                Ok(CudaStorageSlice::U8(out))
            }
        }

        use candle_core::backend::BackendStorage;
        let device = s1.device();
        let slice = self.map(&s1.slice, l1, &s2.slice, l2, device)?;
        Ok((
            CudaStorage {
                slice,
                device: device.clone(),
            },
            l1.shape().clone(),
        ))
    }
}
