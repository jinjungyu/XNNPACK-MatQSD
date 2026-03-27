// MatQSD mqint8 dotprod kernel: single weight, 4-bit/8-bit mode switching.
// Region-separated layout per NR block:
//   [vksum4(64B)] [vksum8(64B)]
//   4-bit region: per group [upper_tiled(c4) | scale_bf16]
//   8-bit extra:  per group [lower_tiled(c4)]
//   [bias(64B)]

#define MQINT8_ROUND_REVERSAL

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

void xnn_qd8_f32_mqint8_gemm_minmax_ukernel_1x16c4__neondot(
    size_t mr, size_t nc, size_t kc,
    const int8_t* restrict a, size_t a_stride,
    const void* restrict w, float* restrict c,
    size_t cm_stride, size_t cn_stride,
    const struct xnn_f32_mqint8_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0 && mr <= 1 && nc != 0 && kc != 0);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const size_t bl = params->scalar.blocksize;
  const int mode = params->scalar.mode;
  const int8x16_t vmask_f0 = vmovq_n_s8((int8_t)0xF0);
  const int8x16_t v8 = vmovq_n_s8(8);
  const float32x4_t v16f = vdupq_n_f32(16.0f);

  do {
    const float32x4_t vzp = vcvtq_f32_s32(
        vld1q_dup_s32(&quantization_params[0].zero_point));

    // Compute NR block index and lower pointer
    const size_t nr_idx =
        ((uintptr_t)w - (uintptr_t)params->scalar.w_base)
        / params->scalar.four_bit_stride;
    const void* wl = (const int8_t*)params->scalar.lower_base
        + nr_idx * params->scalar.lower_per_nr;

    float32x4_t vout0, vout1, vout2, vout3;
    if (mode == 0) {
      // vksum4 from w (4-bit region)
      vout0 = vmulq_f32(vld1q_f32(w), vzp); w = (const float*)w + 4;
      vout1 = vmulq_f32(vld1q_f32(w), vzp); w = (const float*)w + 4;
      vout2 = vmulq_f32(vld1q_f32(w), vzp); w = (const float*)w + 4;
      vout3 = vmulq_f32(vld1q_f32(w), vzp); w = (const float*)w + 4;
    } else {
      // skip vksum4 in w
      w = (const float*)w + 16;
      // vksum8 from wl (lower region)
      vout0 = vmulq_f32(vld1q_f32(wl), vzp); wl = (const float*)wl + 4;
      vout1 = vmulq_f32(vld1q_f32(wl), vzp); wl = (const float*)wl + 4;
      vout2 = vmulq_f32(vld1q_f32(wl), vzp); wl = (const float*)wl + 4;
      vout3 = vmulq_f32(vld1q_f32(wl), vzp); wl = (const float*)wl + 4;
    }

    for (size_t kb = 0; kb < kc; kb += bl) {
      int32x4_t vacc0 = vdupq_n_s32(0);
      int32x4_t vacc1 = vdupq_n_s32(0);
      int32x4_t vacc2 = vdupq_n_s32(0);
      int32x4_t vacc3 = vdupq_n_s32(0);

      if (mode == 0) {
        size_t k = bl;
        while (k >= 8 * sizeof(int8_t)) {
          const int8x8_t va = vld1_s8(a0); a0 += 8;
          const int8x16_t vb0 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb1 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb2 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb3 = vld1q_s8(w); w = (const int8_t*)w + 16;
          vacc0 = vdotq_lane_s32(vacc0, vshlq_n_s8(vb0, 4), va, 0);
          vacc1 = vdotq_lane_s32(vacc1, vshlq_n_s8(vb1, 4), va, 0);
          vacc2 = vdotq_lane_s32(vacc2, vshlq_n_s8(vb2, 4), va, 0);
          vacc3 = vdotq_lane_s32(vacc3, vshlq_n_s8(vb3, 4), va, 0);
          vacc0 = vdotq_lane_s32(vacc0, vandq_s8(vb0, vmask_f0), va, 1);
          vacc1 = vdotq_lane_s32(vacc1, vandq_s8(vb1, vmask_f0), va, 1);
          vacc2 = vdotq_lane_s32(vacc2, vandq_s8(vb2, vmask_f0), va, 1);
          vacc3 = vdotq_lane_s32(vacc3, vandq_s8(vb3, vmask_f0), va, 1);
          k -= 8 * sizeof(int8_t);
        }
        if XNN_UNLIKELY(k != 0) {
          const int8x8_t va = vld1_s8(a0); a0 += 4;
          const int8x16_t vb0 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb1 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb2 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb3 = vld1q_s8(w); w = (const int8_t*)w + 16;
          vacc0 = vdotq_lane_s32(vacc0, vshlq_n_s8(vb0, 4), va, 0);
          vacc1 = vdotq_lane_s32(vacc1, vshlq_n_s8(vb1, 4), va, 0);
          vacc2 = vdotq_lane_s32(vacc2, vshlq_n_s8(vb2, 4), va, 0);
          vacc3 = vdotq_lane_s32(vacc3, vshlq_n_s8(vb3, 4), va, 0);
        }
      } else {
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
        size_t k = bl;
        while (k >= 8 * sizeof(int8_t)) {
          const int8x8_t va = vld1_s8(a0); a0 += 8;
          const int8x16_t vu0 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vu1 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vu2 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vu3 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vl0 = vld1q_s8(wl); wl = (const int8_t*)wl + 16;
          const int8x16_t vl1 = vld1q_s8(wl); wl = (const int8_t*)wl + 16;
          const int8x16_t vl2 = vld1q_s8(wl); wl = (const int8_t*)wl + 16;
          const int8x16_t vl3 = vld1q_s8(wl); wl = (const int8_t*)wl + 16;
          vacc0 = vdotq_lane_s32(vacc0, RECON(vshlq_n_s8(vu0,4), vshlq_n_s8(vl0,4)), va, 0);
          vacc1 = vdotq_lane_s32(vacc1, RECON(vshlq_n_s8(vu1,4), vshlq_n_s8(vl1,4)), va, 0);
          vacc2 = vdotq_lane_s32(vacc2, RECON(vshlq_n_s8(vu2,4), vshlq_n_s8(vl2,4)), va, 0);
          vacc3 = vdotq_lane_s32(vacc3, RECON(vshlq_n_s8(vu3,4), vshlq_n_s8(vl3,4)), va, 0);
          vacc0 = vdotq_lane_s32(vacc0, RECON(vandq_s8(vu0,vmask_f0), vandq_s8(vl0,vmask_f0)), va, 1);
          vacc1 = vdotq_lane_s32(vacc1, RECON(vandq_s8(vu1,vmask_f0), vandq_s8(vl1,vmask_f0)), va, 1);
          vacc2 = vdotq_lane_s32(vacc2, RECON(vandq_s8(vu2,vmask_f0), vandq_s8(vl2,vmask_f0)), va, 1);
          vacc3 = vdotq_lane_s32(vacc3, RECON(vandq_s8(vu3,vmask_f0), vandq_s8(vl3,vmask_f0)), va, 1);
          k -= 8 * sizeof(int8_t);
        }
        if XNN_UNLIKELY(k != 0) {
          const int8x8_t va = vld1_s8(a0); a0 += 4;
          const int8x16_t vu0 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vu1 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vu2 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vu3 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vl0 = vld1q_s8(wl); wl = (const int8_t*)wl + 16;
          const int8x16_t vl1 = vld1q_s8(wl); wl = (const int8_t*)wl + 16;
          const int8x16_t vl2 = vld1q_s8(wl); wl = (const int8_t*)wl + 16;
          const int8x16_t vl3 = vld1q_s8(wl); wl = (const int8_t*)wl + 16;
          vacc0 = vdotq_lane_s32(vacc0, RECON(vshlq_n_s8(vu0,4), vshlq_n_s8(vl0,4)), va, 0);
          vacc1 = vdotq_lane_s32(vacc1, RECON(vshlq_n_s8(vu1,4), vshlq_n_s8(vl1,4)), va, 0);
          vacc2 = vdotq_lane_s32(vacc2, RECON(vshlq_n_s8(vu2,4), vshlq_n_s8(vl2,4)), va, 0);
          vacc3 = vdotq_lane_s32(vacc3, RECON(vshlq_n_s8(vu3,4), vshlq_n_s8(vl3,4)), va, 0);
        }
        #undef RECON
      }

      const float32x4_t vs0 = vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)), v16f); w = (const uint16_t*)w + 4;
      const float32x4_t vs1 = vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)), v16f); w = (const uint16_t*)w + 4;
      const float32x4_t vs2 = vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)), v16f); w = (const uint16_t*)w + 4;
      const float32x4_t vs3 = vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)), v16f); w = (const uint16_t*)w + 4;
      vout0 = vfmaq_f32(vout0, vcvtq_f32_s32(vacc0), vs0);
      vout1 = vfmaq_f32(vout1, vcvtq_f32_s32(vacc1), vs1);
      vout2 = vfmaq_f32(vout2, vcvtq_f32_s32(vacc2), vs2);
      vout3 = vfmaq_f32(vout3, vcvtq_f32_s32(vacc3), vs3);
    }

    // w already points to bias (contiguous after upper+scale in 4-bit region)

    const float32x4_t vinv = vld1q_dup_f32(&quantization_params[0].inv_scale);
    vout0 = vmulq_f32(vout0, vinv); vout1 = vmulq_f32(vout1, vinv);
    vout2 = vmulq_f32(vout2, vinv); vout3 = vmulq_f32(vout3, vinv);
    vout0 = vaddq_f32(vout0, vld1q_f32(w)); w = (const float*)w + 4;
    vout1 = vaddq_f32(vout1, vld1q_f32(w)); w = (const float*)w + 4;
    vout2 = vaddq_f32(vout2, vld1q_f32(w)); w = (const float*)w + 4;
    vout3 = vaddq_f32(vout3, vld1q_f32(w)); w = (const float*)w + 4;

    const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
    const float32x4_t vmax = vdupq_n_f32(params->scalar.max);
    vout0 = vmaxq_f32(vminq_f32(vout0, vmax), vmin);
    vout1 = vmaxq_f32(vminq_f32(vout1, vmax), vmin);
    vout2 = vmaxq_f32(vminq_f32(vout2, vmax), vmin);
    vout3 = vmaxq_f32(vminq_f32(vout3, vmax), vmin);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vout0); vst1q_f32(c0+4, vout1);
      vst1q_f32(c0+8, vout2); vst1q_f32(c0+12, vout3);
      a0 = (const int8_t*)((uintptr_t)a0 - kc);
      c0 = (float*)((uintptr_t)c0 + cn_stride);
      nc -= 16;
    } else {
      if (nc & 8) { vst1q_f32(c0, vout0); c0+=4; vout0=vout2; vst1q_f32(c0, vout1); c0+=4; vout1=vout3; }
      if (nc & 4) { vst1q_f32(c0, vout0); c0+=4; vout0=vout1; }
      float32x2_t vlo = vget_low_f32(vout0);
      if (nc & 2) { vst1_f32(c0, vlo); c0+=2; vlo=vget_high_f32(vout0); }
      if (nc & 1) { vst1_lane_f32(c0, vlo, 0); }
      nc = 0;
    }
  } while (nc != 0);
}
