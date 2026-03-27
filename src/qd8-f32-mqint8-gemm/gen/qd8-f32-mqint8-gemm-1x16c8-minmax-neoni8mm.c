// MatQSD mqint8 i8mm kernel: single weight, 4-bit/8-bit mode switching.
// Region-separated layout per NR block:
//   [vksum4(64B)] [vksum8(64B)]
//   4-bit region: per group [upper_tiled(c8) | scale_bf16]  ← contiguous for 4-bit
//   8-bit extra:  per group [lower_tiled(c8)]
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

void xnn_qd8_f32_mqint8_gemm_minmax_ukernel_1x16c8__neoni8mm(
    size_t mr, size_t nc, size_t kc,
    const int8_t* restrict a, size_t a_stride,
    const void* restrict w, float* restrict c,
    size_t cm_stride, size_t cn_stride,
    const struct xnn_f32_mqint8_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0 && mr <= 1 && nc != 0 && kc != 0);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const size_t bl = params->scalar.blocksize;
  const int mode = params->scalar.mode;

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
    const float32x4_t vzp = vcvtq_f32_s32(
        vld1q_dup_s32(&quantization_params[0].zero_point));

    // Compute NR block index and lower pointer (before consuming vksum)
    const size_t nr_idx =
        ((uintptr_t)w - (uintptr_t)params->scalar.w_base)
        / params->scalar.four_bit_stride;
    const void* wl = (const int8_t*)params->scalar.lower_base
        + nr_idx * params->scalar.lower_per_nr;

    // Load vksum: 4-bit reads vksum4 from w, 8-bit reads vksum8 from wl
    float32x4_t vout0x0123, vout0x4567, vout0x89AB, vout0xCDEF;
    if (mode == 0) {
      // vksum4 is first in 4-bit region
      vout0x0123 = vmulq_f32(vld1q_f32(w), vzp); w = (const float*)w + 4;
      vout0x4567 = vmulq_f32(vld1q_f32(w), vzp); w = (const float*)w + 4;
      vout0x89AB = vmulq_f32(vld1q_f32(w), vzp); w = (const float*)w + 4;
      vout0xCDEF = vmulq_f32(vld1q_f32(w), vzp); w = (const float*)w + 4;
    } else {
      // vksum4 still at w start — skip it
      w = (const float*)w + 16;
      // vksum8 is first in lower region
      vout0x0123 = vmulq_f32(vld1q_f32(wl), vzp); wl = (const float*)wl + 4;
      vout0x4567 = vmulq_f32(vld1q_f32(wl), vzp); wl = (const float*)wl + 4;
      vout0x89AB = vmulq_f32(vld1q_f32(wl), vzp); wl = (const float*)wl + 4;
      vout0xCDEF = vmulq_f32(vld1q_f32(wl), vzp); wl = (const float*)wl + 4;
    }

    for (size_t kb = 0; kb < kc; kb += bl) {
      int32x4_t vacc01x01 = vdupq_n_s32(0);
      int32x4_t vacc01x23 = vdupq_n_s32(0);
      int32x4_t vacc01x45 = vdupq_n_s32(0);
      int32x4_t vacc01x67 = vdupq_n_s32(0);
      int32x4_t vacc01x89 = vdupq_n_s32(0);
      int32x4_t vacc01xAB = vdupq_n_s32(0);
      int32x4_t vacc01xCD = vdupq_n_s32(0);
      int32x4_t vacc01xEF = vdupq_n_s32(0);

      if (mode == 0) {
        // ---- 4-bit: read upper from w (contiguous in 4-bit region) ----
        uint64x2x2_t va01x0123456789ABCDEF;
        va01x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
        va01x0123456789ABCDEF.val[1] = vdupq_n_u64(0);

        size_t k = bl;
        while (k >= 16 * sizeof(int8_t)) {
          va01x0123456789ABCDEF = vld2q_lane_u64((const void*)a0, va01x0123456789ABCDEF, 0); a0 += 16;
          const int8x16_t va01x01234567 = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]);
          const int8x16_t va01x89ABCDEF = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]);

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

          vacc01x01 = vmmlaq_s32(vacc01x01, va01x01234567, vb01_lo);
          vacc01x23 = vmmlaq_s32(vacc01x23, va01x01234567, vb23_lo);
          vacc01x45 = vmmlaq_s32(vacc01x45, va01x01234567, vb45_lo);
          vacc01x67 = vmmlaq_s32(vacc01x67, va01x01234567, vb67_lo);
          vacc01x89 = vmmlaq_s32(vacc01x89, va01x01234567, vb89_lo);
          vacc01xAB = vmmlaq_s32(vacc01xAB, va01x01234567, vbAB_lo);
          vacc01xCD = vmmlaq_s32(vacc01xCD, va01x01234567, vbCD_lo);
          vacc01xEF = vmmlaq_s32(vacc01xEF, va01x01234567, vbEF_lo);
          vacc01x01 = vmmlaq_s32(vacc01x01, va01x89ABCDEF, vb01_hi);
          vacc01x23 = vmmlaq_s32(vacc01x23, va01x89ABCDEF, vb23_hi);
          vacc01x45 = vmmlaq_s32(vacc01x45, va01x89ABCDEF, vb45_hi);
          vacc01x67 = vmmlaq_s32(vacc01x67, va01x89ABCDEF, vb67_hi);
          vacc01x89 = vmmlaq_s32(vacc01x89, va01x89ABCDEF, vb89_hi);
          vacc01xAB = vmmlaq_s32(vacc01xAB, va01x89ABCDEF, vbAB_hi);
          vacc01xCD = vmmlaq_s32(vacc01xCD, va01x89ABCDEF, vbCD_hi);
          vacc01xEF = vmmlaq_s32(vacc01xEF, va01x89ABCDEF, vbEF_hi);

          k -= 16 * sizeof(int8_t);
        }
        if XNN_UNLIKELY(k != 0) {
          uint64x2_t va01x01234567 = vld1q_dup_u64((const void*)a0); a0 += 8;
          const int8x16_t vb01 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb23 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb45 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb67 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vb89 = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vbAB = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vbCD = vld1q_s8(w); w = (const int8_t*)w + 16;
          const int8x16_t vbEF = vld1q_s8(w); w = (const int8_t*)w + 16;
          vacc01x01 = vmmlaq_s32(vacc01x01, vreinterpretq_s8_u64(va01x01234567), vshlq_n_s8(vb01, 4));
          vacc01x23 = vmmlaq_s32(vacc01x23, vreinterpretq_s8_u64(va01x01234567), vshlq_n_s8(vb23, 4));
          vacc01x45 = vmmlaq_s32(vacc01x45, vreinterpretq_s8_u64(va01x01234567), vshlq_n_s8(vb45, 4));
          vacc01x67 = vmmlaq_s32(vacc01x67, vreinterpretq_s8_u64(va01x01234567), vshlq_n_s8(vb67, 4));
          vacc01x89 = vmmlaq_s32(vacc01x89, vreinterpretq_s8_u64(va01x01234567), vshlq_n_s8(vb89, 4));
          vacc01xAB = vmmlaq_s32(vacc01xAB, vreinterpretq_s8_u64(va01x01234567), vshlq_n_s8(vbAB, 4));
          vacc01xCD = vmmlaq_s32(vacc01xCD, vreinterpretq_s8_u64(va01x01234567), vshlq_n_s8(vbCD, 4));
          vacc01xEF = vmmlaq_s32(vacc01xEF, vreinterpretq_s8_u64(va01x01234567), vshlq_n_s8(vbEF, 4));
        }
      } else {
        // ---- 8-bit: read upper from w (4-bit region), lower from wl (8-bit region) ----
        uint64x2x2_t va01x0123456789ABCDEF;
        va01x0123456789ABCDEF.val[0] = vdupq_n_u64(0);
        va01x0123456789ABCDEF.val[1] = vdupq_n_u64(0);

        size_t k = bl;
        while (k >= 16 * sizeof(int8_t)) {
          va01x0123456789ABCDEF = vld2q_lane_u64((const void*)a0, va01x0123456789ABCDEF, 0); a0 += 16;
          const int8x16_t va_lo = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[0]);
          const int8x16_t va_hi = vreinterpretq_s8_u64(va01x0123456789ABCDEF.val[1]);

          #define PROCESS_PAIR(ACC, wu_ptr, wl_ptr) do { \
            const int8x16_t _vu = vld1q_s8(wu_ptr); wu_ptr = (const int8_t*)wu_ptr + 16; \
            const int8x16_t _vl = vld1q_s8(wl_ptr); wl_ptr = (const int8_t*)wl_ptr + 16; \
            ACC = vmmlaq_s32(ACC, va_lo, RECON(vshlq_n_s8(_vu, 4), vshlq_n_s8(_vl, 4))); \
            ACC = vmmlaq_s32(ACC, va_hi, RECON(vandq_s8(_vu, vmask), vandq_s8(_vl, vmask))); \
          } while (0)

          PROCESS_PAIR(vacc01x01, w, wl);
          PROCESS_PAIR(vacc01x23, w, wl);
          PROCESS_PAIR(vacc01x45, w, wl);
          PROCESS_PAIR(vacc01x67, w, wl);
          PROCESS_PAIR(vacc01x89, w, wl);
          PROCESS_PAIR(vacc01xAB, w, wl);
          PROCESS_PAIR(vacc01xCD, w, wl);
          PROCESS_PAIR(vacc01xEF, w, wl);

          k -= 16 * sizeof(int8_t);
        }
        if XNN_UNLIKELY(k != 0) {
          uint64x2_t va01x01234567 = vld1q_dup_u64((const void*)a0); a0 += 8;
          const int8x16_t va_s = vreinterpretq_s8_u64(va01x01234567);

          #define PROCESS_PAIR_TAIL(ACC, wu_ptr, wl_ptr) do { \
            const int8x16_t _vu = vld1q_s8(wu_ptr); wu_ptr = (const int8_t*)wu_ptr + 16; \
            const int8x16_t _vl = vld1q_s8(wl_ptr); wl_ptr = (const int8_t*)wl_ptr + 16; \
            ACC = vmmlaq_s32(ACC, va_s, RECON(vshlq_n_s8(_vu, 4), vshlq_n_s8(_vl, 4))); \
          } while (0)

          PROCESS_PAIR_TAIL(vacc01x01, w, wl);
          PROCESS_PAIR_TAIL(vacc01x23, w, wl);
          PROCESS_PAIR_TAIL(vacc01x45, w, wl);
          PROCESS_PAIR_TAIL(vacc01x67, w, wl);
          PROCESS_PAIR_TAIL(vacc01x89, w, wl);
          PROCESS_PAIR_TAIL(vacc01xAB, w, wl);
          PROCESS_PAIR_TAIL(vacc01xCD, w, wl);
          PROCESS_PAIR_TAIL(vacc01xEF, w, wl);
        }
        #undef PROCESS_PAIR
        #undef PROCESS_PAIR_TAIL
      }

      // Extract row 0 from 2x2 accumulators
      int32x4_t vacc0x0123 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x01), vreinterpretq_u64_s32(vacc01x23)));
      int32x4_t vacc0x4567 = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x45), vreinterpretq_u64_s32(vacc01x67)));
      int32x4_t vacc0x89AB = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01x89), vreinterpretq_u64_s32(vacc01xAB)));
      int32x4_t vacc0xCDEF = vreinterpretq_s32_u64(vtrn1q_u64(vreinterpretq_u64_s32(vacc01xCD), vreinterpretq_u64_s32(vacc01xEF)));

      // Scale ×16 (in 4-bit region, right after upper data)
      const float32x4_t vs0 = vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)), v16f); w = (const uint16_t*)w + 4;
      const float32x4_t vs1 = vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)), v16f); w = (const uint16_t*)w + 4;
      const float32x4_t vs2 = vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)), v16f); w = (const uint16_t*)w + 4;
      const float32x4_t vs3 = vmulq_f32(vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w), 16)), v16f); w = (const uint16_t*)w + 4;
      vout0x0123 = vfmaq_f32(vout0x0123, vcvtq_f32_s32(vacc0x0123), vs0);
      vout0x4567 = vfmaq_f32(vout0x4567, vcvtq_f32_s32(vacc0x4567), vs1);
      vout0x89AB = vfmaq_f32(vout0x89AB, vcvtq_f32_s32(vacc0x89AB), vs2);
      vout0xCDEF = vfmaq_f32(vout0xCDEF, vcvtq_f32_s32(vacc0xCDEF), vs3);
    }

    // After group loop: w points to bias (right after scale in 4-bit region).
    // No skip needed — bias is contiguous after upper+scale data.

    // inv_scale + bias + clamp
    const float32x4_t vinv = vld1q_dup_f32(&quantization_params[0].inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinv);
    vout0x4567 = vmulq_f32(vout0x4567, vinv);
    vout0x89AB = vmulq_f32(vout0x89AB, vinv);
    vout0xCDEF = vmulq_f32(vout0xCDEF, vinv);

    vout0x0123 = vaddq_f32(vout0x0123, vld1q_f32(w)); w = (const float*)w + 4;
    vout0x4567 = vaddq_f32(vout0x4567, vld1q_f32(w)); w = (const float*)w + 4;
    vout0x89AB = vaddq_f32(vout0x89AB, vld1q_f32(w)); w = (const float*)w + 4;
    vout0xCDEF = vaddq_f32(vout0xCDEF, vld1q_f32(w)); w = (const float*)w + 4;

    const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
    const float32x4_t vmax = vdupq_n_f32(params->scalar.max);
    vout0x0123 = vmaxq_f32(vminq_f32(vout0x0123, vmax), vmin);
    vout0x4567 = vmaxq_f32(vminq_f32(vout0x4567, vmax), vmin);
    vout0x89AB = vmaxq_f32(vminq_f32(vout0x89AB, vmax), vmin);
    vout0xCDEF = vmaxq_f32(vminq_f32(vout0xCDEF, vmax), vmin);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);
      vst1q_f32(c0 + 8, vout0x89AB);
      vst1q_f32(c0 + 12, vout0xCDEF);
      a0 = (const int8_t*)((uintptr_t)a0 - kc);
      c0 = (float*)((uintptr_t)c0 + cn_stride);
      nc -= 16;
    } else {
      if (nc & 8) { vst1q_f32(c0, vout0x0123); c0 += 4; vout0x0123 = vout0x89AB; vst1q_f32(c0, vout0x4567); c0 += 4; vout0x4567 = vout0xCDEF; }
      if (nc & 4) { vst1q_f32(c0, vout0x0123); c0 += 4; vout0x0123 = vout0x4567; }
      float32x2_t vlo = vget_low_f32(vout0x0123);
      if (nc & 2) { vst1_f32(c0, vlo); c0 += 2; vlo = vget_high_f32(vout0x0123); }
      if (nc & 1) { vst1_lane_f32(c0, vlo, 0); }
      nc = 0;
    }
  } while (nc != 0);

  #undef RECON
}
