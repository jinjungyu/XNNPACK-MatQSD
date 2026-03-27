// MatQSD mqint8 4x16c8 i8mm GeMM kernel: 4 activation rows × 16 output columns.
// Processes 4 tokens simultaneously for batch verification in speculative decoding.
// Global region separation: 4-bit region (upper+scale+bias), 8-bit region (vksum8+lower).

#define MQINT8_ROUND_REVERSAL

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

void xnn_qd8_f32_mqint8_gemm_minmax_ukernel_4x16c8__neoni8mm(
    size_t mr, size_t nc, size_t kc,
    const int8_t* restrict a, size_t a_stride,
    const void* restrict w, float* restrict c,
    size_t cm_stride, size_t cn_stride,
    const struct xnn_f32_mqint8_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0 && mr <= 4 && nc != 0 && kc != 0);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const size_t bl = params->scalar.blocksize;
  const int mode = params->scalar.mode;

  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*)((uintptr_t)a0 + a_stride);
  float* c1 = (float*)((uintptr_t)c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) { a1 = a0; c1 = c0; }
  const int8_t* a2 = (const int8_t*)((uintptr_t)a1 + a_stride);
  float* c2 = (float*)((uintptr_t)c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) { a2 = a1; c2 = c1; }
  const int8_t* a3 = (const int8_t*)((uintptr_t)a2 + a_stride);
  float* c3 = (float*)((uintptr_t)c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) { a3 = a2; c3 = c2; }

  const int8x16_t vmask = vmovq_n_s8((int8_t)0xF0);
  const int8x16_t v8 = vmovq_n_s8(8);
  const float32x4_t v16f = vdupq_n_f32(16.0f);

#ifdef MQINT8_ROUND_REVERSAL
  #define RECON(vu_s, vl_s) ({ \
    const int8x16_t _lv = vshrq_n_s8(vl_s, 4); \
    const int8x16_t _r = vaddq_s8(vaddq_s8(vu_s, _lv), v8); \
    vsubq_s8(_r, vandq_s8( \
        vreinterpretq_s8_u8(vcgeq_s8(_lv, vdupq_n_s8(0))), \
        vdupq_n_s8(16))); \
  })
#else
  #define RECON(vu_s, vl_s) \
    vaddq_s8(vaddq_s8(vu_s, vshrq_n_s8(vl_s, 4)), v8)
#endif

  do {
    // NR block index and lower pointer (recomputed each NR block)
    const size_t nr_idx =
        ((uintptr_t)w - (uintptr_t)params->scalar.w_base)
        / params->scalar.four_bit_stride;
    const void* wl_base = (const int8_t*)params->scalar.lower_base
        + nr_idx * params->scalar.lower_per_nr;
    // Load vksum for 4 rows
    const float32x4_t vzp01 = vcvtq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    const float32x4_t vzp23 = vcvtq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));

    float32x4_t vout0x0123, vout0x4567, vout0x89AB, vout0xCDEF;
    float32x4_t vout1x0123, vout1x4567, vout1x89AB, vout1xCDEF;
    float32x4_t vout2x0123, vout2x4567, vout2x89AB, vout2xCDEF;
    float32x4_t vout3x0123, vout3x4567, vout3x89AB, vout3xCDEF;

    const void* wl = wl_base;
    if (mode == 0) {
      // vksum4 from w (4-bit region)
      const float32x4_t vk0123 = vld1q_f32(w); w = (const float*)w + 4;
      const float32x4_t vk4567 = vld1q_f32(w); w = (const float*)w + 4;
      const float32x4_t vk89AB = vld1q_f32(w); w = (const float*)w + 4;
      const float32x4_t vkCDEF = vld1q_f32(w); w = (const float*)w + 4;
      vout0x0123 = vmulq_lane_f32(vk0123, vget_low_f32(vzp01), 0);
      vout1x0123 = vmulq_lane_f32(vk0123, vget_high_f32(vzp01), 0);
      vout2x0123 = vmulq_lane_f32(vk0123, vget_low_f32(vzp23), 0);
      vout3x0123 = vmulq_lane_f32(vk0123, vget_high_f32(vzp23), 0);
      vout0x4567 = vmulq_lane_f32(vk4567, vget_low_f32(vzp01), 0);
      vout1x4567 = vmulq_lane_f32(vk4567, vget_high_f32(vzp01), 0);
      vout2x4567 = vmulq_lane_f32(vk4567, vget_low_f32(vzp23), 0);
      vout3x4567 = vmulq_lane_f32(vk4567, vget_high_f32(vzp23), 0);
      vout0x89AB = vmulq_lane_f32(vk89AB, vget_low_f32(vzp01), 0);
      vout1x89AB = vmulq_lane_f32(vk89AB, vget_high_f32(vzp01), 0);
      vout2x89AB = vmulq_lane_f32(vk89AB, vget_low_f32(vzp23), 0);
      vout3x89AB = vmulq_lane_f32(vk89AB, vget_high_f32(vzp23), 0);
      vout0xCDEF = vmulq_lane_f32(vkCDEF, vget_low_f32(vzp01), 0);
      vout1xCDEF = vmulq_lane_f32(vkCDEF, vget_high_f32(vzp01), 0);
      vout2xCDEF = vmulq_lane_f32(vkCDEF, vget_low_f32(vzp23), 0);
      vout3xCDEF = vmulq_lane_f32(vkCDEF, vget_high_f32(vzp23), 0);
    } else {
      // skip vksum4 in w
      w = (const float*)w + 16;
      // vksum8 from wl (lower region)
      const float32x4_t vk0123 = vld1q_f32(wl); wl = (const float*)wl + 4;
      const float32x4_t vk4567 = vld1q_f32(wl); wl = (const float*)wl + 4;
      const float32x4_t vk89AB = vld1q_f32(wl); wl = (const float*)wl + 4;
      const float32x4_t vkCDEF = vld1q_f32(wl); wl = (const float*)wl + 4;
      vout0x0123 = vmulq_lane_f32(vk0123, vget_low_f32(vzp01), 0);
      vout1x0123 = vmulq_lane_f32(vk0123, vget_high_f32(vzp01), 0);
      vout2x0123 = vmulq_lane_f32(vk0123, vget_low_f32(vzp23), 0);
      vout3x0123 = vmulq_lane_f32(vk0123, vget_high_f32(vzp23), 0);
      vout0x4567 = vmulq_lane_f32(vk4567, vget_low_f32(vzp01), 0);
      vout1x4567 = vmulq_lane_f32(vk4567, vget_high_f32(vzp01), 0);
      vout2x4567 = vmulq_lane_f32(vk4567, vget_low_f32(vzp23), 0);
      vout3x4567 = vmulq_lane_f32(vk4567, vget_high_f32(vzp23), 0);
      vout0x89AB = vmulq_lane_f32(vk89AB, vget_low_f32(vzp01), 0);
      vout1x89AB = vmulq_lane_f32(vk89AB, vget_high_f32(vzp01), 0);
      vout2x89AB = vmulq_lane_f32(vk89AB, vget_low_f32(vzp23), 0);
      vout3x89AB = vmulq_lane_f32(vk89AB, vget_high_f32(vzp23), 0);
      vout0xCDEF = vmulq_lane_f32(vkCDEF, vget_low_f32(vzp01), 0);
      vout1xCDEF = vmulq_lane_f32(vkCDEF, vget_high_f32(vzp01), 0);
      vout2xCDEF = vmulq_lane_f32(vkCDEF, vget_low_f32(vzp23), 0);
      vout3xCDEF = vmulq_lane_f32(vkCDEF, vget_high_f32(vzp23), 0);
    }

    for (size_t kb = 0; kb < kc; kb += bl) {
      int32x4_t vacc01x01 = vdupq_n_s32(0), vacc01x23 = vdupq_n_s32(0);
      int32x4_t vacc01x45 = vdupq_n_s32(0), vacc01x67 = vdupq_n_s32(0);
      int32x4_t vacc01x89 = vdupq_n_s32(0), vacc01xAB = vdupq_n_s32(0);
      int32x4_t vacc01xCD = vdupq_n_s32(0), vacc01xEF = vdupq_n_s32(0);
      int32x4_t vacc23x01 = vdupq_n_s32(0), vacc23x23 = vdupq_n_s32(0);
      int32x4_t vacc23x45 = vdupq_n_s32(0), vacc23x67 = vdupq_n_s32(0);
      int32x4_t vacc23x89 = vdupq_n_s32(0), vacc23xAB = vdupq_n_s32(0);
      int32x4_t vacc23xCD = vdupq_n_s32(0), vacc23xEF = vdupq_n_s32(0);

      uint64x2x2_t va01 = {vdupq_n_u64(0), vdupq_n_u64(0)};
      uint64x2x2_t va23 = {vdupq_n_u64(0), vdupq_n_u64(0)};

      if (mode == 0) {
        // ---- 4-bit: read upper, nibble extract ----
        size_t k = bl;
        while (k >= 16 * sizeof(int8_t)) {
          va01 = vld2q_lane_u64((const void*)a0, va01, 0); a0 += 16;
          va23 = vld2q_lane_u64((const void*)a2, va23, 0); a2 += 16;
          va01 = vld2q_lane_u64((const void*)a1, va01, 1); a1 += 16;
          va23 = vld2q_lane_u64((const void*)a3, va23, 1); a3 += 16;
          const int8x16_t va01_lo = vreinterpretq_s8_u64(va01.val[0]);
          const int8x16_t va01_hi = vreinterpretq_s8_u64(va01.val[1]);
          const int8x16_t va23_lo = vreinterpretq_s8_u64(va23.val[0]);
          const int8x16_t va23_hi = vreinterpretq_s8_u64(va23.val[1]);

          // Load 8 weight vectors + pre-split nibbles (matching qb4w pattern)
          const int8x16_t vb01 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb23 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb45 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb67 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb89 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vbAB = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vbCD = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vbEF = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb01_lo = vshlq_n_s8(vb01, 4);
          const int8x16_t vb23_lo = vshlq_n_s8(vb23, 4);
          const int8x16_t vb45_lo = vshlq_n_s8(vb45, 4);
          const int8x16_t vb67_lo = vshlq_n_s8(vb67, 4);
          const int8x16_t vb89_lo = vshlq_n_s8(vb89, 4);
          const int8x16_t vbAB_lo = vshlq_n_s8(vbAB, 4);
          const int8x16_t vbCD_lo = vshlq_n_s8(vbCD, 4);
          const int8x16_t vbEF_lo = vshlq_n_s8(vbEF, 4);
          const int8x16_t vb01_hi = vandq_s8(vb01, vmask);
          const int8x16_t vb23_hi = vandq_s8(vb23, vmask);
          const int8x16_t vb45_hi = vandq_s8(vb45, vmask);
          const int8x16_t vb67_hi = vandq_s8(vb67, vmask);
          const int8x16_t vb89_hi = vandq_s8(vb89, vmask);
          const int8x16_t vbAB_hi = vandq_s8(vbAB, vmask);
          const int8x16_t vbCD_hi = vandq_s8(vbCD, vmask);
          const int8x16_t vbEF_hi = vandq_s8(vbEF, vmask);

          // rows 01 × lo nibbles, rows 23 × lo nibbles (interleaved, reuse weight)
          vacc01x01 = vmmlaq_s32(vacc01x01, va01_lo, vb01_lo);
          vacc01x23 = vmmlaq_s32(vacc01x23, va01_lo, vb23_lo);
          vacc01x45 = vmmlaq_s32(vacc01x45, va01_lo, vb45_lo);
          vacc01x67 = vmmlaq_s32(vacc01x67, va01_lo, vb67_lo);
          vacc01x89 = vmmlaq_s32(vacc01x89, va01_lo, vb89_lo);
          vacc01xAB = vmmlaq_s32(vacc01xAB, va01_lo, vbAB_lo);
          vacc01xCD = vmmlaq_s32(vacc01xCD, va01_lo, vbCD_lo);
          vacc01xEF = vmmlaq_s32(vacc01xEF, va01_lo, vbEF_lo);
          vacc23x01 = vmmlaq_s32(vacc23x01, va23_lo, vb01_lo);
          vacc23x23 = vmmlaq_s32(vacc23x23, va23_lo, vb23_lo);
          vacc23x45 = vmmlaq_s32(vacc23x45, va23_lo, vb45_lo);
          vacc23x67 = vmmlaq_s32(vacc23x67, va23_lo, vb67_lo);
          vacc23x89 = vmmlaq_s32(vacc23x89, va23_lo, vb89_lo);
          vacc23xAB = vmmlaq_s32(vacc23xAB, va23_lo, vbAB_lo);
          vacc23xCD = vmmlaq_s32(vacc23xCD, va23_lo, vbCD_lo);
          vacc23xEF = vmmlaq_s32(vacc23xEF, va23_lo, vbEF_lo);
          // rows 01 × hi nibbles, rows 23 × hi nibbles
          vacc01x01 = vmmlaq_s32(vacc01x01, va01_hi, vb01_hi);
          vacc01x23 = vmmlaq_s32(vacc01x23, va01_hi, vb23_hi);
          vacc01x45 = vmmlaq_s32(vacc01x45, va01_hi, vb45_hi);
          vacc01x67 = vmmlaq_s32(vacc01x67, va01_hi, vb67_hi);
          vacc01x89 = vmmlaq_s32(vacc01x89, va01_hi, vb89_hi);
          vacc01xAB = vmmlaq_s32(vacc01xAB, va01_hi, vbAB_hi);
          vacc01xCD = vmmlaq_s32(vacc01xCD, va01_hi, vbCD_hi);
          vacc01xEF = vmmlaq_s32(vacc01xEF, va01_hi, vbEF_hi);
          vacc23x01 = vmmlaq_s32(vacc23x01, va23_hi, vb01_hi);
          vacc23x23 = vmmlaq_s32(vacc23x23, va23_hi, vb23_hi);
          vacc23x45 = vmmlaq_s32(vacc23x45, va23_hi, vb45_hi);
          vacc23x67 = vmmlaq_s32(vacc23x67, va23_hi, vb67_hi);
          vacc23x89 = vmmlaq_s32(vacc23x89, va23_hi, vb89_hi);
          vacc23xAB = vmmlaq_s32(vacc23xAB, va23_hi, vbAB_hi);
          vacc23xCD = vmmlaq_s32(vacc23xCD, va23_hi, vbCD_hi);
          vacc23xEF = vmmlaq_s32(vacc23xEF, va23_hi, vbEF_hi);
          k -= 16 * sizeof(int8_t);
        }
        if XNN_UNLIKELY(k != 0) {
          uint64x2_t va01t = vld1q_dup_u64((const void*)a0); a0 += 8;
          uint64x2_t va23t = vld1q_dup_u64((const void*)a2); a2 += 8;
          va01t = vld1q_lane_u64((const void*)a1, va01t, 1); a1 += 8;
          va23t = vld1q_lane_u64((const void*)a3, va23t, 1); a3 += 8;
          const int8x16_t vb01 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb23 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb45 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb67 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb89 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vbAB = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vbCD = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vbEF = vld1q_s8(w); w = (const int8_t*)w + 16;
          #define DO_TAIL(ACC, va, vb) ACC = vmmlaq_s32(ACC, vreinterpretq_s8_u64(va), vshlq_n_s8(vb, 4))
          DO_TAIL(vacc01x01, va01t, vb01); DO_TAIL(vacc01x23, va01t, vb23);
          DO_TAIL(vacc01x45, va01t, vb45); DO_TAIL(vacc01x67, va01t, vb67);
          DO_TAIL(vacc01x89, va01t, vb89); DO_TAIL(vacc01xAB, va01t, vbAB);
          DO_TAIL(vacc01xCD, va01t, vbCD); DO_TAIL(vacc01xEF, va01t, vbEF);
          DO_TAIL(vacc23x01, va23t, vb01); DO_TAIL(vacc23x23, va23t, vb23);
          DO_TAIL(vacc23x45, va23t, vb45); DO_TAIL(vacc23x67, va23t, vb67);
          DO_TAIL(vacc23x89, va23t, vb89); DO_TAIL(vacc23xAB, va23t, vbAB);
          DO_TAIL(vacc23xCD, va23t, vbCD); DO_TAIL(vacc23xEF, va23t, vbEF);
          #undef DO_TAIL
        }
      } else {
        // ---- 8-bit: read upper from w, lower from wl, reconstruct ----
        size_t k = bl;
        while (k >= 16 * sizeof(int8_t)) {
          va01 = vld2q_lane_u64((const void*)a0, va01, 0); a0 += 16;
          va23 = vld2q_lane_u64((const void*)a2, va23, 0); a2 += 16;
          va01 = vld2q_lane_u64((const void*)a1, va01, 1); a1 += 16;
          va23 = vld2q_lane_u64((const void*)a3, va23, 1); a3 += 16;
          const int8x16_t va01_lo = vreinterpretq_s8_u64(va01.val[0]);
          const int8x16_t va01_hi = vreinterpretq_s8_u64(va01.val[1]);
          const int8x16_t va23_lo = vreinterpretq_s8_u64(va23.val[0]);
          const int8x16_t va23_hi = vreinterpretq_s8_u64(va23.val[1]);

          // Load upper(w) + lower(wl), RECON both nibble halves, vmmlaq for both row pairs
          #define PAIR(ACC01, ACC23) do { \
            const int8x16_t _vu = vld1q_s8(w);  w  = (const int8_t*)w  + 16; \
            const int8x16_t _vl = vld1q_s8(wl); wl = (const int8_t*)wl + 16; \
            const int8x16_t _rlo = RECON(vshlq_n_s8(_vu,4), vshlq_n_s8(_vl,4)); \
            const int8x16_t _rhi = RECON(vandq_s8(_vu,vmask), vandq_s8(_vl,vmask)); \
            ACC01 = vmmlaq_s32(ACC01, va01_lo, _rlo); \
            ACC23 = vmmlaq_s32(ACC23, va23_lo, _rlo); \
            ACC01 = vmmlaq_s32(ACC01, va01_hi, _rhi); \
            ACC23 = vmmlaq_s32(ACC23, va23_hi, _rhi); \
          } while(0)

          PAIR(vacc01x01, vacc23x01);
          PAIR(vacc01x23, vacc23x23);
          PAIR(vacc01x45, vacc23x45);
          PAIR(vacc01x67, vacc23x67);
          PAIR(vacc01x89, vacc23x89);
          PAIR(vacc01xAB, vacc23xAB);
          PAIR(vacc01xCD, vacc23xCD);
          PAIR(vacc01xEF, vacc23xEF);
          #undef PAIR
          k -= 16 * sizeof(int8_t);
        }
        if XNN_UNLIKELY(k != 0) {
          uint64x2_t va01t = vld1q_dup_u64((const void*)a0); a0 += 8;
          uint64x2_t va23t = vld1q_dup_u64((const void*)a2); a2 += 8;
          va01t = vld1q_lane_u64((const void*)a1, va01t, 1); a1 += 8;
          va23t = vld1q_lane_u64((const void*)a3, va23t, 1); a3 += 8;
          #define PAIR_TAIL(ACC01, ACC23) do { \
            const int8x16_t _vu = vld1q_s8(w);  w  = (const int8_t*)w  + 16; \
            const int8x16_t _vl = vld1q_s8(wl); wl = (const int8_t*)wl + 16; \
            const int8x16_t _r = RECON(vshlq_n_s8(_vu,4), vshlq_n_s8(_vl,4)); \
            ACC01 = vmmlaq_s32(ACC01, vreinterpretq_s8_u64(va01t), _r); \
            ACC23 = vmmlaq_s32(ACC23, vreinterpretq_s8_u64(va23t), _r); \
          } while(0)
          PAIR_TAIL(vacc01x01, vacc23x01); PAIR_TAIL(vacc01x23, vacc23x23);
          PAIR_TAIL(vacc01x45, vacc23x45); PAIR_TAIL(vacc01x67, vacc23x67);
          PAIR_TAIL(vacc01x89, vacc23x89); PAIR_TAIL(vacc01xAB, vacc23xAB);
          PAIR_TAIL(vacc01xCD, vacc23xCD); PAIR_TAIL(vacc01xEF, vacc23xEF);
          #undef PAIR_TAIL
        }
      }

      // Extract 4 rows from 2x2 accumulators
      #define EX(r, ACC, a, b) int32x4_t vacc##r = vreinterpretq_s32_u64(vtrn##a##q_u64(vreinterpretq_u64_s32(ACC), vreinterpretq_u64_s32(b)))
      EX(0x0123, vacc01x01, 1, vacc01x23); EX(1x0123, vacc01x01, 2, vacc01x23);
      EX(0x4567, vacc01x45, 1, vacc01x67); EX(1x4567, vacc01x45, 2, vacc01x67);
      EX(0x89AB, vacc01x89, 1, vacc01xAB); EX(1x89AB, vacc01x89, 2, vacc01xAB);
      EX(0xCDEF, vacc01xCD, 1, vacc01xEF); EX(1xCDEF, vacc01xCD, 2, vacc01xEF);
      EX(2x0123, vacc23x01, 1, vacc23x23); EX(3x0123, vacc23x01, 2, vacc23x23);
      EX(2x4567, vacc23x45, 1, vacc23x67); EX(3x4567, vacc23x45, 2, vacc23x67);
      EX(2x89AB, vacc23x89, 1, vacc23xAB); EX(3x89AB, vacc23x89, 2, vacc23xAB);
      EX(2xCDEF, vacc23xCD, 1, vacc23xEF); EX(3xCDEF, vacc23xCD, 2, vacc23xEF);
      #undef EX

      // Scale ×16
      const float32x4_t vs0 = vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)), v16f); w = (const uint16_t*)w + 4;
      const float32x4_t vs1 = vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)), v16f); w = (const uint16_t*)w + 4;
      const float32x4_t vs2 = vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)), v16f); w = (const uint16_t*)w + 4;
      const float32x4_t vs3 = vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)), v16f); w = (const uint16_t*)w + 4;

      #define FMA(r, s) \
        vout##r##x0123 = vfmaq_f32(vout##r##x0123, vcvtq_f32_s32(vacc##r##x0123), s##0); \
        vout##r##x4567 = vfmaq_f32(vout##r##x4567, vcvtq_f32_s32(vacc##r##x4567), s##1); \
        vout##r##x89AB = vfmaq_f32(vout##r##x89AB, vcvtq_f32_s32(vacc##r##x89AB), s##2); \
        vout##r##xCDEF = vfmaq_f32(vout##r##xCDEF, vcvtq_f32_s32(vacc##r##xCDEF), s##3)
      FMA(0, vs); FMA(1, vs); FMA(2, vs); FMA(3, vs);
      #undef FMA
    }

    // inv_scale per row
    const float32x4_t vis01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    const float32x4_t vis23 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    #define SCALE_ROW(r, vis, lane) \
      vout##r##x0123 = vmulq_lane_f32(vout##r##x0123, vget_##lane##_f32(vis), 1); \
      vout##r##x4567 = vmulq_lane_f32(vout##r##x4567, vget_##lane##_f32(vis), 1); \
      vout##r##x89AB = vmulq_lane_f32(vout##r##x89AB, vget_##lane##_f32(vis), 1); \
      vout##r##xCDEF = vmulq_lane_f32(vout##r##xCDEF, vget_##lane##_f32(vis), 1)
    SCALE_ROW(0, vis01, low); SCALE_ROW(1, vis01, high);
    SCALE_ROW(2, vis23, low); SCALE_ROW(3, vis23, high);
    #undef SCALE_ROW

    // bias (shared across rows)
    const float32x4_t vb0 = vld1q_f32(w); w = (const float*)w + 4;
    const float32x4_t vb1 = vld1q_f32(w); w = (const float*)w + 4;
    const float32x4_t vb2 = vld1q_f32(w); w = (const float*)w + 4;
    const float32x4_t vb3 = vld1q_f32(w); w = (const float*)w + 4;
    #define ADD_BIAS(r) \
      vout##r##x0123 = vaddq_f32(vout##r##x0123, vb0); \
      vout##r##x4567 = vaddq_f32(vout##r##x4567, vb1); \
      vout##r##x89AB = vaddq_f32(vout##r##x89AB, vb2); \
      vout##r##xCDEF = vaddq_f32(vout##r##xCDEF, vb3)
    ADD_BIAS(0); ADD_BIAS(1); ADD_BIAS(2); ADD_BIAS(3);
    #undef ADD_BIAS

    // clamp
    const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
    const float32x4_t vmax = vdupq_n_f32(params->scalar.max);
    #define CLAMP(r) \
      vout##r##x0123 = vmaxq_f32(vminq_f32(vout##r##x0123, vmax), vmin); \
      vout##r##x4567 = vmaxq_f32(vminq_f32(vout##r##x4567, vmax), vmin); \
      vout##r##x89AB = vmaxq_f32(vminq_f32(vout##r##x89AB, vmax), vmin); \
      vout##r##xCDEF = vmaxq_f32(vminq_f32(vout##r##xCDEF, vmax), vmin)
    CLAMP(0); CLAMP(1); CLAMP(2); CLAMP(3);
    #undef CLAMP

    if XNN_LIKELY(nc >= 16) {
      #define STORE(r) vst1q_f32(c##r, vout##r##x0123); vst1q_f32(c##r+4, vout##r##x4567); \
                       vst1q_f32(c##r+8, vout##r##x89AB); vst1q_f32(c##r+12, vout##r##xCDEF)
      STORE(0); STORE(1); STORE(2); STORE(3);
      #undef STORE
      a0 = (const int8_t*)((uintptr_t)a0 - kc); a1 = (const int8_t*)((uintptr_t)a1 - kc);
      a2 = (const int8_t*)((uintptr_t)a2 - kc); a3 = (const int8_t*)((uintptr_t)a3 - kc);
      c0 = (float*)((uintptr_t)c0 + cn_stride); c1 = (float*)((uintptr_t)c1 + cn_stride);
      c2 = (float*)((uintptr_t)c2 + cn_stride); c3 = (float*)((uintptr_t)c3 + cn_stride);
      nc -= 16;
    } else {
      #define TAIL_STORE(op, r) op(c##r, vout##r##x0123); op(c##r, vout##r##x4567); \
                                op(c##r, vout##r##x89AB); op(c##r, vout##r##xCDEF)
      if (nc & 8) {
        #define S8(r) vst1q_f32(c##r, vout##r##x0123); c##r+=4; vout##r##x0123=vout##r##x89AB; \
                      vst1q_f32(c##r, vout##r##x4567); c##r+=4; vout##r##x4567=vout##r##xCDEF
        S8(0); S8(1); S8(2); S8(3);
        #undef S8
      }
      if (nc & 4) {
        #define S4(r) vst1q_f32(c##r, vout##r##x0123); c##r+=4; vout##r##x0123=vout##r##x4567
        S4(0); S4(1); S4(2); S4(3);
        #undef S4
      }
      float32x2_t vlo0=vget_low_f32(vout0x0123), vlo1=vget_low_f32(vout1x0123);
      float32x2_t vlo2=vget_low_f32(vout2x0123), vlo3=vget_low_f32(vout3x0123);
      if (nc & 2) {
        vst1_f32(c0,vlo0);c0+=2; vst1_f32(c1,vlo1);c1+=2;
        vst1_f32(c2,vlo2);c2+=2; vst1_f32(c3,vlo3);c3+=2;
        vlo0=vget_high_f32(vout0x0123); vlo1=vget_high_f32(vout1x0123);
        vlo2=vget_high_f32(vout2x0123); vlo3=vget_high_f32(vout3x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c0,vlo0,0); vst1_lane_f32(c1,vlo1,0);
        vst1_lane_f32(c2,vlo2,0); vst1_lane_f32(c3,vlo3,0);
      }
      #undef TAIL_STORE
      nc = 0;
    }
  } while (nc != 0);

  #undef RECON
}
