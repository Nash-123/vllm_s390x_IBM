#ifndef CPU_TYPES_VXE_HPP
#define CPU_TYPES_VXE_HPP

#include <vecintrin.h>
#include <cmath>
#include <torch/all.h>

namespace vec_op {

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)                                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef CPU_OP_GUARD
#define CPU_KERNEL_GUARD_IN(NAME)
#define CPU_KERNEL_GUARD_OUT(NAME)
#else
#define CPU_KERNEL_GUARD_IN(NAME)                                              \
  std::cout << #NAME << " invoked." << std::endl;
#define CPU_KERNEL_GUARD_OUT(NAME) std::cout << #NAME << " exit." << std::endl;
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline

#define vec_neg(a) (-(a))
#define vec_add(a, b) ((a) + (b))
#define vec_sub(a, b) ((a) - (b))
#define vec_mul(a, b) ((a) * (b))
#define vec_div(a, b) ((a) / (b))

const static __vector unsigned char omask = {
  0, 1, 4, 5,
  8, 9, 12, 13,
  16, 17, 20, 21,
  24, 25, 28, 29
};

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F &&f) {
  (f(std::integral_constant<T, indexes>{}), ...);
}
}; // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F &&f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T> struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; }
};

typedef struct ss16x8x2_t {
  __vector signed short val[2];
} ss16x8x2_t;

typedef struct ss16x8x4_t {
  __vector signed short val[4];
} ss16x8x4_t;

typedef struct f32x4x2_t {
  __vector float val[2];
} f32x4x2_t;

typedef struct f32x4x4_t {
  __vector float val[4];
} f32x4x4_t;

struct FP32Vec8;
struct FP32Vec16;

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  __vector signed short reg;

  explicit BF16Vec8(const void *ptr) {
    // Load considering big-endian
    reg = *(__vector signed short*)ptr;
    reg = (__vector signed short)vec_perm(reg, reg, omask); // Apply permutation for big-endian
  }

  explicit BF16Vec8(const FP32Vec8 &);

  void save(void *ptr) const {
    __vector signed short temp = reg;
    temp = (__vector signed short)vec_perm(temp, temp, omask); // Apply permutation before saving
    *reinterpret_cast<__vector signed short *>(ptr) = temp;
  }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  ss16x8x2_t reg;

  explicit BF16Vec16(const void *ptr) {
    // Load 256 bits in two parts and apply big-endian conversion
    reg.val[0] = (__vector signed short)vec_xl(0,  (signed short *)ptr);
    reg.val[1] = (__vector signed short)vec_xl(16, (signed short *)ptr);
    reg.val[0] = (__vector signed short)vec_perm(reg.val[0], reg.val[0], omask);
    reg.val[1] = (__vector signed short)vec_perm(reg.val[1], reg.val[1], omask);
  }

  explicit BF16Vec16(const FP32Vec16 &);

  void save(void *ptr) const {
    // Save 256 bits in two parts with big-endian conversion
    __vector signed short temp0 = reg.val[0];
    __vector signed short temp1 = reg.val[1];
    temp0 = (__vector signed short)vec_perm(temp0, temp0, omask);
    temp1 = (__vector signed short)vec_perm(temp1, temp1, omask);
    vec_xst(temp0, 0, (signed short *)ptr);
    vec_xst(temp1, 16, (signed short *)ptr);
  }
};

const static __vector signed short zero = vec_splats((signed short)0);

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;

  ss16x8x4_t reg;

  explicit BF16Vec32(const void *ptr) {
    reg = *reinterpret_cast<const ss16x8x4_t *>(ptr);
    reg.val[0] = (__vector signed short)vec_perm(reg.val[0], reg.val[0], omask);
    reg.val[1] = (__vector signed short)vec_perm(reg.val[1], reg.val[1], omask);
    reg.val[2] = (__vector signed short)vec_perm(reg.val[2], reg.val[2], omask);
    reg.val[3] = (__vector signed short)vec_perm(reg.val[3], reg.val[3], omask);
  }

  explicit BF16Vec32(ss16x8x4_t data) : reg(data) {}

  explicit BF16Vec32(const BF16Vec8 &vec8_data) : reg({
    vec8_data.reg,
    vec8_data.reg,
    vec8_data.reg,
    vec8_data.reg
  }) {}

  void save(void *ptr) const { *reinterpret_cast<ss16x8x4_t *>(ptr) = reg; }
};

struct FP32Vec4 : public Vec<FP32Vec4> {
  constexpr static int VEC_ELEM_NUM = 4;
  union AliasReg {
    __vector float reg;
    float values[VEC_ELEM_NUM];
  };

  __vector float reg;

  explicit FP32Vec4(float v) : reg(vec_splats(v)) {}

  explicit FP32Vec4() : reg(vec_splats(0.0f)) {}

  explicit FP32Vec4(const float *ptr) : reg(vec_xl(0, ptr)) {}

  explicit FP32Vec4(__vector float data) : reg(data) {}

  explicit FP32Vec4(const FP32Vec4 &data) : reg(data.reg) {}
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  union AliasReg {
    f32x4x2_t reg;
    float values[VEC_ELEM_NUM];
  };

  f32x4x2_t reg;

  explicit FP32Vec8(float v) {
    reg.val[0] = vec_splats(v);
    reg.val[1] = vec_splats(v);
  }

  explicit FP32Vec8() {
    reg.val[0] = vec_splats(0.0f);
    reg.val[1] = vec_splats(0.0f);
  }

  explicit FP32Vec8(const float *ptr) {
    reg.val[0] = vec_xl(0, ptr);
    reg.val[1] = vec_xl(16, ptr);
    reg.val[0] = (__vector float)vec_perm(reg.val[0], reg.val[0], omask); // Apply permutation
    reg.val[1] = (__vector float)vec_perm(reg.val[1], reg.val[1], omask);
  }

  explicit FP32Vec8(f32x4x2_t data) : reg(data) {}

  explicit FP32Vec8(const FP32Vec8 &data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[1];
  }

  explicit FP32Vec8(const BF16Vec8 &v) {
    reg.val[0] = (__vector float)vec_mergeh(zero, v.reg);
    reg.val[1] = (__vector float)vec_mergel(zero, v.reg);
  }

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float result = 0;
    unroll_loop<int, VEC_ELEM_NUM>([&result, &ar](int i) { result += ar.values[i]; });
    return result;
  }

FP32Vec8 exp() const {
    AliasReg ar;
    ar.reg = reg;

    f32x4x2_t ret;

    // Compute exponentials in big-endian format
    ret.val[0][0] = std::exp(ar.values[3]); // Big-endian: reverse order for lower half
    ret.val[0][1] = std::exp(ar.values[2]);
    ret.val[0][2] = std::exp(ar.values[1]);
    ret.val[0][3] = std::exp(ar.values[0]);

    ret.val[1][0] = std::exp(ar.values[7]); // Big-endian: reverse order for upper half
    ret.val[1][1] = std::exp(ar.values[6]);
    ret.val[1][2] = std::exp(ar.values[5]);
    ret.val[1][3] = std::exp(ar.values[4]);

    return FP32Vec8(f32x4x2_t({ret.val[0], ret.val[1]}));
}

FP32Vec8 tanh() const {
    AliasReg ar;
    ar.reg = reg;
    f32x4x4_t ret;
    ret.val[0][0] = std::tanh(ar.values[3]); // Reverse order for big-endian
    ret.val[0][1] = std::tanh(ar.values[2]);
    ret.val[0][2] = std::tanh(ar.values[1]);
    ret.val[0][3] = std::tanh(ar.values[0]);
    ret.val[1][0] = std::tanh(ar.values[7]);
    ret.val[1][1] = std::tanh(ar.values[6]);
    ret.val[1][2] = std::tanh(ar.values[5]);
    ret.val[1][3] = std::tanh(ar.values[4]);
    return FP32Vec8(f32x4x2_t({ret.val[0], ret.val[1]}));
}

FP32Vec8 er() const {
    AliasReg ar;
    ar.reg = reg;
    f32x4x4_t ret;
    ret.val[0][0] = std::erf(ar.values[3]); // Reverse order for big-endian
    ret.val[0][1] = std::erf(ar.values[2]);
    ret.val[0][2] = std::erf(ar.values[1]);
    ret.val[0][3] = std::erf(ar.values[0]);
    ret.val[1][0] = std::erf(ar.values[7]);
    ret.val[1][1] = std::erf(ar.values[6]);
    ret.val[1][2] = std::erf(ar.values[5]);
    ret.val[1][3] = std::erf(ar.values[4]);
    return FP32Vec8(f32x4x2_t({ret.val[0], ret.val[1]}));
}

FP32Vec8 operator*(const FP32Vec8 &b) const {
    return FP32Vec8({
        vec_mul(b.reg.val[0], reg.val[0]), // Reverse order for big-endian
        vec_mul(b.reg.val[1], reg.val[1])
    });
}

FP32Vec8 operator+(const FP32Vec8 &b) const {
    return FP32Vec8({
        vec_add(b.reg.val[0], reg.val[0]), // Reverse order for big-endian
        vec_add(b.reg.val[1], reg.val[1])
    });
}

FP32Vec8 operator-(const FP32Vec8 &b) const {
    return FP32Vec8({
        vec_sub(b.reg.val[0], reg.val[0]), // Reverse order for big-endian
        vec_sub(b.reg.val[1], reg.val[1])
    });
}

FP32Vec8 operator/(const FP32Vec8 &b) const {
    return FP32Vec8({
        vec_div(b.reg.val[0], reg.val[0]), // Reverse order for big-endian
        vec_div(b.reg.val[1], reg.val[1])
    });
}

void save(float *ptr) const {
    vec_xst(reg.val[1], 0, ptr); // Save high part first
    vec_xst(reg.val[0], 16, ptr); // Then save low part
}
};
struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    f32x4x4_t reg;
    float values[VEC_ELEM_NUM];
  };

  f32x4x4_t reg;

  explicit FP32Vec16(float v) {
    reg.val[0] = vec_splats(v);
    reg.val[1] = vec_splats(v);
    reg.val[2] = vec_splats(v);
    reg.val[3] = vec_splats(v);
  }

  explicit FP32Vec16() {
    reg.val[0] = vec_splats(0.0f);
    reg.val[1] = vec_splats(0.0f);
    reg.val[2] = vec_splats(0.0f);
    reg.val[3] = vec_splats(0.0f);
  }

  explicit FP32Vec16(const float *ptr) {
    // Load in big-endian order
    reg.val[3] = vec_xl(0, ptr);   // Highest part first
    reg.val[2] = vec_xl(16, ptr);
    reg.val[1] = vec_xl(32, ptr);
    reg.val[0] = vec_xl(48, ptr);  // Lowest part last
  }

  explicit FP32Vec16(f32x4x4_t data) : reg(data) {}

  explicit FP32Vec16(const FP32Vec16 &data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[1];
    reg.val[2] = data.reg.val[2];
    reg.val[3] = data.reg.val[3];
  }

  explicit FP32Vec16(const FP32Vec4 &data) {
    reg.val[0] = data.reg;
    reg.val[1] = data.reg;
    reg.val[2] = data.reg;
    reg.val[3] = data.reg;
  }

  explicit FP32Vec16(const FP32Vec8 &data) {
    reg.val[0] = data.reg.val[1]; // Adjusted for big-endian
    reg.val[1] = data.reg.val[0]; // Adjusted for big-endian
    reg.val[2] = data.reg.val[1]; // Adjusted for big-endian
    reg.val[3] = data.reg.val[0]; // Adjusted for big-endian
  }

  explicit FP32Vec16(const BF16Vec16 &v) {
    reg.val[0] = (__vector float)vec_mergeh(zero, v.reg.val[1]); // Adjusted for big-endian
    reg.val[1] = (__vector float)vec_mergel(zero, v.reg.val[1]); // Adjusted for big-endian
    reg.val[2] = (__vector float)vec_mergeh(zero, v.reg.val[0]); // Adjusted for big-endian
    reg.val[3] = (__vector float)vec_mergel(zero, v.reg.val[0]); // Adjusted for big-endian
  }

  explicit FP32Vec16(const BF16Vec8 &v) : FP32Vec16(FP32Vec8(v)) {}

  FP32Vec16 operator*(const FP32Vec16 &b) const {
    return FP32Vec16(f32x4x4_t({
        vec_mul(reg.val[3], b.reg.val[3]), // Adjusted for big-endian
        vec_mul(reg.val[2], b.reg.val[2]),
        vec_mul(reg.val[1], b.reg.val[1]),
        vec_mul(reg.val[0], b.reg.val[0])})); // Adjusted for big-endian
  }

  FP32Vec16 operator+(const FP32Vec16 &b) const {
    return FP32Vec16(f32x4x4_t({
        vec_add(reg.val[3], b.reg.val[3]), // Adjusted for big-endian
        vec_add(reg.val[2], b.reg.val[2]),
        vec_add(reg.val[1], b.reg.val[1]),
        vec_add(reg.val[0], b.reg.val[0])})); // Adjusted for big-endian
  }

  FP32Vec16 operator-(const FP32Vec16 &b) const {
    return FP32Vec16(f32x4x4_t({
        vec_sub(reg.val[3], b.reg.val[3]), // Adjusted for big-endian
        vec_sub(reg.val[2], b.reg.val[2]),
        vec_sub(reg.val[1], b.reg.val[1]),
        vec_sub(reg.val[0], b.reg.val[0])})); // Adjusted for big-endian
  }

  FP32Vec16 operator/(const FP32Vec16 &b) const {
    return FP32Vec16(f32x4x4_t({
        vec_div(reg.val[3], b.reg.val[3]), // Adjusted for big-endian
        vec_div(reg.val[2], b.reg.val[2]),
        vec_div(reg.val[1], b.reg.val[1]),
        vec_div(reg.val[0], b.reg.val[0])})); // Adjusted for big-endian
  }

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float result = 0;
    unroll_loop<int, VEC_ELEM_NUM>([&result, &ar](int i) { result += ar.values[i]; }); // Access in normal order

    return result;
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);

    AliasReg ar;
    ar.reg = reg;
    float result = 0;
    const int start = idx * group_size; // Adjust start for big-endian
    unroll_loop<int, group_size>(
        [&result, &start, ar](int i) { result += ar.values[start + i]; }); // Access in normal order

    return result;
  }

  void save(float *ptr) const {
    // Save in big-endian order
    vec_xst(reg.val[3], 0, ptr); // Save highest part first
    vec_xst(reg.val[2], 16, ptr);
    vec_xst(reg.val[1], 32, ptr);
    vec_xst(reg.val[0], 48, ptr); // Save lowest part last
  }
};

template <typename T> struct VecType { using vec_type = void; };

template <typename T> using vec_t = typename VecType<T>::vec_type;

// Specializations for vector types
template <> struct VecType<float> { using vec_type = FP32Vec8; };
template <> struct VecType<c10::BFloat16> { using vec_type = BF16Vec8; };

// Store float value into a pointer
template <typename T> void storeFP32(float v, T *ptr) { *ptr = v; }

// Fused multiply-add operation
inline void fma(FP32Vec16 &acc, FP32Vec16 &a, FP32Vec16 &b) {
  acc = acc + a * b;
}

// Specialized storage for BFloat16
template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16 *ptr) {
  c10::BFloat16 __attribute__((__may_alias__)) *v_ptr = reinterpret_cast<c10::BFloat16 *>(&v);
  *ptr = *(v_ptr + 1); // Store the high part of the float
}

// Mask for specific vector operations
#ifndef __VEC_CLASS_FP_NAN
#define __VEC_CLASS_FP_NAN (1 << 6)
#endif

inline BF16Vec8::BF16Vec8(const FP32Vec8 &v) {
#ifdef _ARCH_S390X
  __vector unsigned int inp0 = (__vector unsigned int)(v.reg.val[0]);
  __vector unsigned int inp1 = (__vector unsigned int)(v.reg.val[1]);

  // Extract LSBs from the original vectors (the lower 16 bits)
  __vector unsigned int lsb0 = vec_and(inp0, mask16); // mask16 isolates the lower 16 bits
  __vector unsigned int lsb1 = vec_and(inp1, mask16);

  // Rounding logic
  __vector unsigned int rnd0 = vec_add(lsb0, bias);
  __vector unsigned int rnd1 = vec_add(lsb1, bias);
  inp0 = vec_add(inp0, rnd0);
  inp1 = vec_add(inp1, rnd1);

  // Handle NaN values
  __vector __bool int sel0 = vec_test_data_class(v.reg.val[0], __VEC_CLASS_FP_NAN);
  __vector __bool int sel1 = vec_test_data_class(v.reg.val[1], __VEC_CLASS_FP_NAN);
  inp0 = vec_sel(inp0, nan, sel0);
  inp1 = vec_sel(inp1, nan, sel1);

  // Shift right to isolate the BF16 result (after rounding)
  inp0 = vec_sr(inp0, sh16);
  inp1 = vec_sr(inp1, sh16);

  // Combine the two results for BF16 storage
  reg = (__vector signed short)vec_perm(inp0, inp1, omask);
#endif
}

inline BF16Vec16::BF16Vec16(const FP32Vec16 &v) {
#ifdef _ARCH_S390X
  __vector unsigned int inp0 = (__vector unsigned int)(v.reg.val[0]);
  __vector unsigned int inp1 = (__vector unsigned int)(v.reg.val[1]);
  __vector unsigned int inp2 = (__vector unsigned int)(v.reg.val[2]);
  __vector unsigned int inp3 = (__vector unsigned int)(v.reg.val[3]);

  // Extract the least significant bits (LSBs) for rounding
  __vector unsigned int lsb0 = vec_sr(inp0, sh16);
  __vector unsigned int lsb1 = vec_sr(inp1, sh16);
  __vector unsigned int lsb2 = vec_sr(inp2, sh16);
  __vector unsigned int lsb3 = vec_sr(inp3, sh16);

  lsb0 = vec_and(lsb0, one);
  lsb1 = vec_and(lsb1, one);
  lsb2 = vec_and(lsb2, one);
  lsb3 = vec_and(lsb3, one);

  // Rounding logic
  __vector unsigned int rnd0 = vec_add(lsb0, bias);
  __vector unsigned int rnd1 = vec_add(lsb1, bias);
  __vector unsigned int rnd2 = vec_add(lsb2, bias);
  __vector unsigned int rnd3 = vec_add(lsb3, bias);

  inp0 = vec_add(inp0, rnd0);
  inp1 = vec_add(inp1, rnd1);
  inp2 = vec_add(inp2, rnd2);
  inp3 = vec_add(inp3, rnd3);

  // Handle NaN values
  __vector __bool int sel0 = vec_test_data_class(v.reg.val[0], __VEC_CLASS_FP_NAN);
  __vector __bool int sel1 = vec_test_data_class(v.reg.val[1], __VEC_CLASS_FP_NAN);
  __vector __bool int sel2 = vec_test_data_class(v.reg.val[2], __VEC_CLASS_FP_NAN);
  __vector __bool int sel3 = vec_test_data_class(v.reg.val[3], __VEC_CLASS_FP_NAN);

  inp0 = vec_sel(inp0, nan, sel0);
  inp1 = vec_sel(inp1, nan, sel1);
  inp2 = vec_sel(inp2, nan, sel2);
  inp3 = vec_sel(inp3, nan, sel3);

  // Final shift to prepare for BF16 representation
  inp0 = vec_sr(inp0, sh16);
  inp1 = vec_sr(inp1, sh16);
  inp2 = vec_sr(inp2, sh16);
  inp3 = vec_sr(inp3, sh16);

  // Store the results in the correct order for big-endian
  reg.val[0] = (__vector signed short)vec_perm(inp0, inp1, omask);
  reg.val[1] = (__vector signed short)vec_perm(inp2, inp3, omask);

  // You may need to continue for additional input values if necessary
  // Ensure to handle all vectors in pairs as required by your design
#endif
}

inline void prefetch(const void *addr) {
  __asm__ __volatile__("");
}

}; // namespace vec_op

#endif
