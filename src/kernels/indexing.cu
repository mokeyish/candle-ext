#include "compatibility.cuh"
#include<stdint.h>

#define SCATTER_OP(TYPENAME, INDEX_TYPENAME, FN_NAME, OP) \
extern "C" __global__ void FN_NAME(  \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t dst_dim_size, \
    const size_t right_size \
) { \
    const size_t numel = left_size * right_size;\
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {\
          const size_t pre = i / right_size;\
          const size_t post = i % right_size;\
          for (unsigned int j = 0; j < src_dim_size; ++j) {\
              const size_t src_i = (pre * src_dim_size + j) * right_size + post;\
              const size_t idx = ids[src_i];\
              const size_t dst_i = (pre * dst_dim_size + idx) * right_size + post;\
              out[dst_i] OP inp[src_i];\
          }\
      }\
 } \

#if __CUDA_ARCH__ >= 800
SCATTER_OP(__nv_bfloat16, int64_t, scatter_i64_bf16, =)
SCATTER_OP(__nv_bfloat16, uint32_t, scatter_u32_bf16, =)
SCATTER_OP(__nv_bfloat16, uint8_t, scatter_u8_bf16, =)
#endif

#if __CUDA_ARCH__ >= 530
SCATTER_OP(__half, uint32_t, scatter_u32_f16, =)
SCATTER_OP(__half, uint8_t, scatter_u8_f16, =)
#endif


#pragma region scatter_assign
SCATTER_OP(float, int64_t, scatter_i64_f32, =)
SCATTER_OP(double, int64_t, scatter_i64_f64, =)
SCATTER_OP(uint8_t, int64_t, scatter_i64_u8, =)
SCATTER_OP(int64_t, int64_t, scatter_i64_i64, =)
SCATTER_OP(uint32_t, int64_t, scatter_i64_u32, =)

SCATTER_OP(float, uint32_t, scatter_u32_f32, =)
SCATTER_OP(double, uint32_t, scatter_u32_f64, =)
SCATTER_OP(uint8_t, uint32_t, scatter_u32_u8, =)
SCATTER_OP(int64_t, uint32_t, scatter_u32_i64, =)
SCATTER_OP(uint32_t, uint32_t, scatter_u32_u32, =)

SCATTER_OP(float, uint8_t, scatter_u8_f32, =)
SCATTER_OP(double, uint8_t, scatter_u8_f64, =)
SCATTER_OP(uint8_t, uint8_t, scatter_u8_u8, =)
SCATTER_OP(uint32_t, uint8_t, scatter_u8_u32, =)
SCATTER_OP(int64_t, uint8_t, scatter_u8_i64, =)
#pragma endregion scatter_assign