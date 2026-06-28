//! AVX-512-BF16 (x86_64): hardware `bf16`↔`f32` conversion and the `vdpbf16ps` dot-product.
//!
//! All **stable** intrinsics — the `__m*bh` carrier types and these ops stabilized in 1.89, so
//! nothing here needs nightly. `bf16` has no native ALU (the ISA has only convert + dot), so
//! element-wise compute still happens in `f32`; these calls remove the *software* `bf16`↔`f32`
//! round-trip at the load/store boundary, and [`dp`] is the packed multiply-accumulate the matmul
//! fast path builds on. Reached only where the CPU implements `avx512bf16` (Cooper Lake+, Zen 4+).
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use half::bf16;

/// Widen 16 contiguous `bf16` (unaligned) to `f32x16`.
#[target_feature(enable = "avx512bf16,avx512f,avx512bw")]
#[inline]
pub unsafe fn widen(p: *const bf16) -> __m512 {
    let raw: __m256i = core::ptr::read_unaligned(p as *const __m256i);
    _mm512_cvtpbh_ps(core::mem::transmute::<__m256i, __m256bh>(raw))
}

/// Narrow `f32x16` to 16 contiguous `bf16` (unaligned), round-to-nearest-even in hardware.
#[target_feature(enable = "avx512bf16,avx512f,avx512bw")]
#[inline]
pub unsafe fn narrow(v: __m512, p: *mut bf16) {
    let packed: __m256bh = _mm512_cvtneps_pbh(v);
    core::ptr::write_unaligned(p as *mut __m256i, core::mem::transmute::<__m256bh, __m256i>(packed));
}

/// VNNI-pack two K-rows of a 16-wide column block: result lane `n` holds `lo[n]` in its low 16
/// bits and `hi[n]` in its high 16 bits. Built from `vpmovzxwd` (a defined element→lane mapping,
/// so the layout is correct by construction — no cross-lane shuffle to get wrong).
#[target_feature(enable = "avx512bf16,avx512f,avx512bw")]
#[inline]
pub unsafe fn pack_pair(lo: *const bf16, hi: *const bf16) -> __m512i {
    let l: __m256i = core::ptr::read_unaligned(lo as *const __m256i);
    let h: __m256i = core::ptr::read_unaligned(hi as *const __m256i);
    _mm512_or_si512(
        _mm512_cvtepu16_epi32(l),
        _mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(h)),
    )
}

/// Broadcast one A-pair to every lane: each 32-bit lane holds `a0` (low 16) and `a1` (high 16).
#[target_feature(enable = "avx512bf16,avx512f,avx512bw")]
#[inline]
pub unsafe fn bcast_pair(a0: bf16, a1: bf16) -> __m512i {
    _mm512_set1_epi32((((a1.to_bits() as u32) << 16) | a0.to_bits() as u32) as i32)
}

/// `acc[n] += a[n].lo·b[n].lo + a[n].hi·b[n].hi` lane-wise (`vdpbf16ps`): the bf16 products are
/// exact in f32, then accumulated. Pair `a` from [`bcast_pair`] with `b` from [`pack_pair`].
#[target_feature(enable = "avx512bf16,avx512f,avx512bw")]
#[inline]
pub unsafe fn dp(acc: __m512, a: __m512i, b: __m512i) -> __m512 {
    _mm512_dpbf16_ps(
        acc,
        core::mem::transmute::<__m512i, __m512bh>(a),
        core::mem::transmute::<__m512i, __m512bh>(b),
    )
}
