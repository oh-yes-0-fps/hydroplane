//! The "uniform" scalar element abstraction.
//!
//! A [`Scalar`] is the smallest float element a kernel operates on: `f32`, `f64`,
//! and (with the `half` feature) `f16`/`bf16`. Each scalar declares a [`Scalar::Compute`]
//! type — the precision its math is actually carried out in. For `f32`/`f64` that is the
//! type itself; for `f16`/`bf16`, which have no useful native arithmetic on most targets,
//! `Compute = f32` and the scalar ops widen-compute-narrow. Keeping that policy *here*
//! (not in each backend) is what makes the scalar (1-lane) backend a faithful oracle for
//! the widening SIMD path.

// The CPU's scalar square-root instruction, reached through `core::arch` so it works without `std`
// (where `f32::sqrt` is unavailable). `sqrtss`/`sqrtsd` on x86-64 (SSE/SSE2 are baseline there);
// `fsqrt` on aarch64 (mandatory in the base FP ISA). IEEE correctly-rounded — bit-identical to
// `std`'s `x.sqrt()`, which lowers to the same instruction.
#[cfg(all(target_arch = "x86_64", any(test, not(feature = "std"))))]
#[inline(always)]
fn hw_sqrt_f32(x: f32) -> f32 {
    use core::arch::x86_64::{_mm_cvtss_f32, _mm_set_ss, _mm_sqrt_ss};
    // SAFETY: SSE is part of the x86-64 baseline, so the intrinsic is always available.
    unsafe { _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))) }
}

#[cfg(all(target_arch = "x86_64", any(test, not(feature = "std"))))]
#[inline(always)]
fn hw_sqrt_f64(x: f64) -> f64 {
    use core::arch::x86_64::{_mm_cvtsd_f64, _mm_set_sd, _mm_sqrt_sd};
    // SAFETY: SSE2 is part of the x86-64 baseline.
    unsafe { _mm_cvtsd_f64(_mm_sqrt_sd(_mm_set_sd(x), _mm_set_sd(x))) }
}

#[cfg(all(target_arch = "aarch64", any(test, not(feature = "std"))))]
#[inline(always)]
fn hw_sqrt_f32(x: f32) -> f32 {
    let r: f32;
    // SAFETY: `fsqrt` is a pure data op (no memory, no flags); `s` selects the 32-bit view of a V reg.
    unsafe {
        core::arch::asm!("fsqrt {y:s}, {x:s}", x = in(vreg) x, y = out(vreg) r, options(pure, nomem, nostack));
    }
    r
}

#[cfg(all(target_arch = "aarch64", any(test, not(feature = "std"))))]
#[inline(always)]
fn hw_sqrt_f64(x: f64) -> f64 {
    let r: f64;
    // SAFETY: as `hw_sqrt_f32`; `d` selects the 64-bit view.
    unsafe {
        core::arch::asm!("fsqrt {y:d}, {x:d}", x = in(vreg) x, y = out(vreg) r, options(pure, nomem, nostack));
    }
    r
}

/// Scalar `sqrt`. `std` lowers `x.sqrt()` to the hardware instruction; without `std` we reach that
/// same instruction through [`hw_sqrt_f32`] on x86-64/aarch64. Only an exotic no-`std` target with no
/// hardware sqrt falls back to `libm`, or — last resort — a software Newton loop.
#[inline(always)]
fn sqrt_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(all(not(feature = "std"), any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        hw_sqrt_f32(x)
    }
    #[cfg(all(
        not(feature = "std"),
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        feature = "libm"
    ))]
    {
        libm::sqrtf(x)
    }
    #[cfg(all(
        not(feature = "std"),
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        not(feature = "libm")
    ))]
    {
        software_sqrt_f32(x)
    }
}

#[inline(always)]
fn sqrt_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(all(not(feature = "std"), any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        hw_sqrt_f64(x)
    }
    #[cfg(all(
        not(feature = "std"),
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        feature = "libm"
    ))]
    {
        libm::sqrt(x)
    }
    #[cfg(all(
        not(feature = "std"),
        not(any(target_arch = "x86_64", target_arch = "aarch64")),
        not(feature = "libm")
    ))]
    {
        software_sqrt_f64(x)
    }
}

// Portable Newton fallback — only for a no-`std`, no-`libm` build on a target without a hardware
// sqrt (the SPIR-V backend lowers `sqrt` via spirv-std and never reaches here). `NaN` for negatives,
// matching the hardware; `0`/`inf` pass straight through.
#[cfg(all(
    not(feature = "std"),
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    not(feature = "libm")
))]
#[inline]
fn software_sqrt_f32(x: f32) -> f32 {
    if x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let mut g = x;
    let mut i = 0;
    while i < 20 {
        g = 0.5 * (g + x / g);
        i += 1;
    }
    g
}

#[cfg(all(
    not(feature = "std"),
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    not(feature = "libm")
))]
#[inline]
fn software_sqrt_f64(x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let mut g = x;
    let mut i = 0;
    while i < 24 {
        g = 0.5 * (g + x / g);
        i += 1;
    }
    g
}

/// A scalar float element a kernel can be generic over.
///
/// All arithmetic is expressed as inherent methods (rather than `core::ops`) so that
/// `f16`/`bf16` can transparently route through [`Scalar::Compute`] without the operator
/// traits implying native half-precision math.
pub trait Scalar: Copy + PartialEq + PartialOrd + Send + Sync + 'static {
    /// The precision this element's math is carried out in.
    /// `f32 -> f32`, `f64 -> f64`, `f16/bf16 -> f32`.
    type Compute: Scalar<Compute = Self::Compute>;

    const ZERO: Self;
    const ONE: Self;

    /// Build from an `f64` literal (for in-kernel constants like `4.0`).
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;

    /// Widen to the compute precision and back. No-ops for `f32`/`f64`.
    fn widen(self) -> Self::Compute;
    fn narrow(c: Self::Compute) -> Self;

    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn div(self, o: Self) -> Self;
    fn neg(self) -> Self;
    fn sqrt(self) -> Self;
    fn min(self, o: Self) -> Self;
    fn max(self, o: Self) -> Self;

    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        // Default: not a true fused op; backends with hardware FMA override.
        self.mul(b).add(c)
    }

    #[inline(always)]
    fn le(self, o: Self) -> bool {
        self.partial_cmp(&o)
            .is_some_and(|c| c != core::cmp::Ordering::Greater)
    }
    #[inline(always)]
    fn lt(self, o: Self) -> bool {
        self.partial_cmp(&o) == Some(core::cmp::Ordering::Less)
    }
    #[inline(always)]
    fn ge(self, o: Self) -> bool {
        self.partial_cmp(&o)
            .is_some_and(|c| c != core::cmp::Ordering::Less)
    }
    #[inline(always)]
    fn gt(self, o: Self) -> bool {
        self.partial_cmp(&o) == Some(core::cmp::Ordering::Greater)
    }
}

impl Scalar for f32 {
    type Compute = f32;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn widen(self) -> f32 {
        self
    }
    #[inline(always)]
    fn narrow(c: f32) -> Self {
        c
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        self + o
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        self - o
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        self * o
    }
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        self / o
    }
    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        sqrt_f32(self)
    }
    #[inline(always)]
    fn min(self, o: Self) -> Self {
        // match SIMD min semantics (non-propagating, second-on-NaN is backend-specific;
        // we use a plain comparison so the scalar oracle is deterministic).
        if o < self { o } else { self }
    }
    #[inline(always)]
    fn max(self, o: Self) -> Self {
        if o > self { o } else { self }
    }
}

impl Scalar for f64 {
    type Compute = f64;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn widen(self) -> f64 {
        self
    }
    #[inline(always)]
    fn narrow(c: f64) -> Self {
        c
    }
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        self + o
    }
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        self - o
    }
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        self * o
    }
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        self / o
    }
    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        sqrt_f64(self)
    }
    #[inline(always)]
    fn min(self, o: Self) -> Self {
        if o < self { o } else { self }
    }
    #[inline(always)]
    fn max(self, o: Self) -> Self {
        if o > self { o } else { self }
    }
}

mod half_impls {
    use super::Scalar;
    use half::{bf16, f16};

    macro_rules! impl_half {
        ($ty:ident) => {
            impl Scalar for $ty {
                // Math happens in f32 — `half` provides no SIMD arithmetic, and most
                // targets have no native f16/bf16 ALU. Storage stays 16-bit.
                type Compute = f32;
                const ZERO: Self = $ty::from_f32_const(0.0);
                const ONE: Self = $ty::from_f32_const(1.0);
                #[inline(always)]
                fn from_f64(v: f64) -> Self {
                    $ty::from_f64(v)
                }
                #[inline(always)]
                fn to_f64(self) -> f64 {
                    self.to_f64()
                }
                #[inline(always)]
                fn widen(self) -> f32 {
                    self.to_f32()
                }
                #[inline(always)]
                fn narrow(c: f32) -> Self {
                    $ty::from_f32(c)
                }
                #[inline(always)]
                fn add(self, o: Self) -> Self {
                    $ty::from_f32(self.to_f32() + o.to_f32())
                }
                #[inline(always)]
                fn sub(self, o: Self) -> Self {
                    $ty::from_f32(self.to_f32() - o.to_f32())
                }
                #[inline(always)]
                fn mul(self, o: Self) -> Self {
                    $ty::from_f32(self.to_f32() * o.to_f32())
                }
                #[inline(always)]
                fn div(self, o: Self) -> Self {
                    $ty::from_f32(self.to_f32() / o.to_f32())
                }
                #[inline(always)]
                fn neg(self) -> Self {
                    $ty::from_f32(-self.to_f32())
                }
                #[inline(always)]
                fn sqrt(self) -> Self {
                    $ty::from_f32(Scalar::sqrt(self.to_f32()))
                }
                #[inline(always)]
                fn min(self, o: Self) -> Self {
                    if o.to_f32() < self.to_f32() { o } else { self }
                }
                #[inline(always)]
                fn max(self, o: Self) -> Self {
                    if o.to_f32() > self.to_f32() { o } else { self }
                }
            }
        };
    }

    impl_half!(f16);
    impl_half!(bf16);
}

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "aarch64")))]
mod hw_sqrt_tests {
    #[test]
    fn matches_std_bit_for_bit() {
        let xs32 = [
            0.0f32, 1.0, 2.0, 3.0, 4.0, 9.0, 0.25, 1e-12, 1e12, 123.456, f32::MIN_POSITIVE, f32::MAX,
        ];
        for &x in &xs32 {
            assert_eq!(super::hw_sqrt_f32(x).to_bits(), x.sqrt().to_bits(), "f32 sqrt({x})");
        }
        let xs64 = [
            0.0f64, 1.0, 2.0, 3.0, 4.0, 9.0, 0.25, 1e-300, 1e300, 123.456, f64::MIN_POSITIVE, f64::MAX,
        ];
        for &x in &xs64 {
            assert_eq!(super::hw_sqrt_f64(x).to_bits(), x.sqrt().to_bits(), "f64 sqrt({x})");
        }
        assert!(super::hw_sqrt_f32(-1.0).is_nan());
        assert!(super::hw_sqrt_f64(-1.0).is_nan());
        assert_eq!(super::hw_sqrt_f32(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(super::hw_sqrt_f64(-0.0).to_bits(), (-0.0f64).to_bits());
    }
}
