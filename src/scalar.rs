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

/// A scalar element a kernel can be generic over — the family-neutral core shared by the
/// float elements (`f32`/`f64`/`f16`/`bf16`, see [`FloatScalar`]) and the integer elements
/// (`u32`/`i32`, see [`IntScalar`]). It carries exactly what the generic gang machinery needs:
/// wrapping-or-IEEE arithmetic via the operator bounds, ordering, identities, `f64` literal
/// bridging, the 32-bit pattern bridge, and family-correct `min`/`max`/`abs`/`neg` (floats:
/// IEEE 754-2019 minimumNumber and sign ops; integers: `Ord` and wrapping ops).
pub trait Scalar:
    Copy
    + PartialEq
    + PartialOrd
    + Send
    + Sync
    + 'static
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;

    /// Build from an `f64` literal (for in-kernel constants like `4.0`). Truncating for the
    /// integer elements.
    fn from_f64(v: f64) -> Self;
    fn into_f64(self) -> f64;

    /// The lane's bit pattern in the low bits of a `u32` — the scalar half of the 32-bit
    /// integer-companion bridge ([`Backend::to_bits`](crate::Backend::to_bits)). Exact for
    /// 32-bit elements; `f16`/`bf16` zero-extend their 16 bits; `f64` truncates to the low half
    /// of its pattern (lossy — bit tricks are 32-bit-element territory).
    fn to_bits32(self) -> u32;
    /// Inverse of [`to_bits32`](Scalar::to_bits32); for `f64` the high pattern half is zeroed.
    fn from_bits32(v: u32) -> Self;

    /// Lane arithmetic under the backend contract: IEEE for the float elements, wrapping for
    /// the integer elements (SIMD integer semantics — no debug-overflow panics).
    fn wadd(self, o: Self) -> Self;
    fn wsub(self, o: Self) -> Self;
    fn wmul(self, o: Self) -> Self;

    /// Family-correct minimum: IEEE 754-2019 minimumNumber for floats (one NaN operand yields
    /// the other), `Ord` for integers.
    fn min(self, o: Self) -> Self;
    fn max(self, o: Self) -> Self;
    /// Family-correct absolute value: IEEE sign clear for floats (NaN stays NaN), wrapping for
    /// signed integers (`abs(i32::MIN) == i32::MIN`), identity for unsigned.
    fn abs(self) -> Self;
    /// Family-correct negation: IEEE sign flip for floats, wrapping for integers.
    fn neg(self) -> Self;
}

/// A float element: everything in [`Scalar`], plus IEEE classification/constants (via
/// [`num_traits::float::FloatCore`] — `core`-only, so every no-`std` configuration keeps them),
/// division, the compute-precision widening policy, `sqrt` (reached without `std` via the
/// hardware instruction), and `fma`.
pub trait FloatScalar:
    Scalar + num_traits::float::FloatCore + core::ops::Div<Output = Self> + core::ops::Neg<Output = Self>
{
    /// The precision this element's math is carried out in.
    /// `f32 -> f32`, `f64 -> f64`, `f16/bf16 -> f32`.
    type Compute: FloatScalar<Compute = Self::Compute>;

    /// Widen to the compute precision and back. No-ops for `f32`/`f64`.
    fn widen(self) -> Self::Compute;
    fn narrow(c: Self::Compute) -> Self;

    fn sqrt(self) -> Self;

    #[inline(always)]
    fn fma(self, b: Self, c: Self) -> Self {
        // Default: not a true fused op; backends with hardware FMA override.
        self * b + c
    }
}

/// An integer element: everything in [`Scalar`], plus the bit-manipulation surface via
/// [`num_traits::PrimInt`] (shifts, bitwise ops, counting). Arithmetic on integer varyings is
/// wrapping, matching SIMD integer instructions; there is deliberately no vector division (no
/// ISA has one).
pub trait IntScalar: Scalar + num_traits::PrimInt {}

macro_rules! impl_float_core_scalar {
    ($ty:ident) => {
        #[inline(always)]
        fn wadd(self, o: Self) -> Self {
            self + o
        }
        #[inline(always)]
        fn wsub(self, o: Self) -> Self {
            self - o
        }
        #[inline(always)]
        fn wmul(self, o: Self) -> Self {
            self * o
        }
        #[inline(always)]
        fn min(self, o: Self) -> Self {
            // IEEE minimumNumber, matching the SIMD backends' contract.
            if self.is_nan() {
                o
            } else if o.is_nan() {
                self
            } else if o < self {
                o
            } else {
                self
            }
        }
        #[inline(always)]
        fn max(self, o: Self) -> Self {
            if self.is_nan() {
                o
            } else if o.is_nan() {
                self
            } else if o > self {
                o
            } else {
                self
            }
        }
        #[inline(always)]
        fn abs(self) -> Self {
            <$ty as num_traits::float::FloatCore>::abs(self)
        }
        #[inline(always)]
        fn neg(self) -> Self {
            -self
        }
    };
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline(always)]
    fn into_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_bits32(self) -> u32 {
        self.to_bits()
    }
    #[inline(always)]
    fn from_bits32(v: u32) -> Self {
        f32::from_bits(v)
    }
    impl_float_core_scalar!(f32);
}

impl FloatScalar for f32 {
    type Compute = f32;
    #[inline(always)]
    fn widen(self) -> f32 {
        self
    }
    #[inline(always)]
    fn narrow(c: f32) -> Self {
        c
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        sqrt_f32(self)
    }
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn into_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn to_bits32(self) -> u32 {
        self.to_bits() as u32
    }
    #[inline(always)]
    fn from_bits32(v: u32) -> Self {
        f64::from_bits(v as u64)
    }
    impl_float_core_scalar!(f64);
}

impl FloatScalar for f64 {
    type Compute = f64;
    #[inline(always)]
    fn widen(self) -> f64 {
        self
    }
    #[inline(always)]
    fn narrow(c: f64) -> Self {
        c
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        sqrt_f64(self)
    }
}

macro_rules! impl_int_scalar {
    ($ty:ident, $abs:expr) => {
        impl Scalar for $ty {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                v as $ty
            }
            #[inline(always)]
            fn into_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn to_bits32(self) -> u32 {
                self as u32
            }
            #[inline(always)]
            fn from_bits32(v: u32) -> Self {
                v as $ty
            }
            #[inline(always)]
            fn wadd(self, o: Self) -> Self {
                self.wrapping_add(o)
            }
            #[inline(always)]
            fn wsub(self, o: Self) -> Self {
                self.wrapping_sub(o)
            }
            #[inline(always)]
            fn wmul(self, o: Self) -> Self {
                self.wrapping_mul(o)
            }
            #[inline(always)]
            fn min(self, o: Self) -> Self {
                Ord::min(self, o)
            }
            #[inline(always)]
            fn max(self, o: Self) -> Self {
                Ord::max(self, o)
            }
            #[inline(always)]
            fn abs(self) -> Self {
                #[allow(clippy::redundant_closure_call)]
                ($abs)(self)
            }
            #[inline(always)]
            fn neg(self) -> Self {
                self.wrapping_neg()
            }
        }

        impl IntScalar for $ty {}
    };
}

impl_int_scalar!(u32, |x| x);
impl_int_scalar!(i32, |x: i32| x.wrapping_abs());

mod half_impls {
    use super::{FloatScalar, Scalar};
    use half::{bf16, f16};

    macro_rules! impl_half {
        ($ty:ident) => {
            impl Scalar for $ty {
                const ZERO: Self = $ty::from_f32_const(0.0);
                const ONE: Self = $ty::from_f32_const(1.0);
                #[inline(always)]
                fn from_f64(v: f64) -> Self {
                    $ty::from_f64(v)
                }
                #[inline(always)]
                fn into_f64(self) -> f64 {
                    $ty::to_f64(self)
                }
                #[inline(always)]
                fn to_bits32(self) -> u32 {
                    self.to_bits() as u32
                }
                #[inline(always)]
                fn from_bits32(v: u32) -> Self {
                    $ty::from_bits(v as u16)
                }
                #[inline(always)]
                fn wadd(self, o: Self) -> Self {
                    self + o
                }
                #[inline(always)]
                fn wsub(self, o: Self) -> Self {
                    self - o
                }
                #[inline(always)]
                fn wmul(self, o: Self) -> Self {
                    self * o
                }
                #[inline(always)]
                fn min(self, o: Self) -> Self {
                    if self.is_nan() {
                        o
                    } else if o.is_nan() {
                        self
                    } else if o.to_f32() < self.to_f32() {
                        o
                    } else {
                        self
                    }
                }
                #[inline(always)]
                fn max(self, o: Self) -> Self {
                    if self.is_nan() {
                        o
                    } else if o.is_nan() {
                        self
                    } else if o.to_f32() > self.to_f32() {
                        o
                    } else {
                        self
                    }
                }
                #[inline(always)]
                fn abs(self) -> Self {
                    $ty::from_bits(self.to_bits() & 0x7fff)
                }
                #[inline(always)]
                fn neg(self) -> Self {
                    -self
                }
            }

            // Math happens in f32 — `half` provides no SIMD arithmetic, and most targets have
            // no native f16/bf16 ALU. Storage stays 16-bit; the operator and `FloatCore` impls
            // from `half` widen-compute-narrow the same way.
            impl FloatScalar for $ty {
                type Compute = f32;
                #[inline(always)]
                fn widen(self) -> f32 {
                    self.to_f32()
                }
                #[inline(always)]
                fn narrow(c: f32) -> Self {
                    $ty::from_f32(c)
                }
                #[inline(always)]
                fn sqrt(self) -> Self {
                    $ty::from_f32(FloatScalar::sqrt(self.to_f32()))
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
