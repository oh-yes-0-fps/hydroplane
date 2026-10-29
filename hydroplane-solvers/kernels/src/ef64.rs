//! Emulated `f64` for GPUs without it (Apple's): an unevaluated sum `hi + lo` of two `f32`s with
//! `|lo| ≤ ulp(hi) / 2`, about 48 significant bits in the same 8 bytes as an `f64`.
//!
//! Sums are error-free transformations and products recover their rounding error with a fused
//! multiply-add, so every operation depends on the compiler neither reassociating, fusing nor
//! dropping a rounding step. Three things keep it that way:
//! - no error-free sum has a constant operand, which `spirv-opt` (run by rust-gpu) would fold
//!   (`(c + x) − c → x`);
//! - products are formed by [`prod`], never as a multiplication a compiler could fuse into the
//!   sum that recovers its error;
//! - the host loads a copy of the module with every operation decorated `NoContraction` (which
//!   rust-gpu does not emit), as NVIDIA's compiler needs, except on MoltenVK, where SPIRV-Cross
//!   turns the decoration into unoptimized calls and Metal with fast math off needs none.

use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Ef64 {
    pub hi: f32,
    pub lo: f32,
}

/// `a · b`, rounded once. Bit for bit a multiplication (`x + −0 = x` for every `x`), but a fused
/// multiply-add already, so no driver contracts it into the addition that consumes it.
#[inline(always)]
fn prod(a: f32, b: f32) -> f32 {
    a.mul_add(b, -0.0)
}

/// `a + b` and its rounding error, exactly.
#[inline(always)]
pub fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// [`two_sum`] for `|a| ≥ |b|`.
#[inline(always)]
fn quick_two_sum(a: f32, b: f32) -> Ef64 {
    let s = a + b;
    Ef64 { hi: s, lo: b - (s - a) }
}

impl Ef64 {
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };

    #[inline(always)]
    pub fn from_f32(v: f32) -> Self {
        Self { hi: v, lo: 0.0 }
    }

    /// `self · b + c` with one renormalization instead of a product's and a sum's: the error
    /// is that of a sum of the operands' magnitudes, as in `f64` without cancellation.
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        let p = prod(self.hi, b.hi);
        let e = self.hi.mul_add(b.hi, -p);
        let e = self.hi.mul_add(b.lo, self.lo.mul_add(b.hi, e));
        let (s, t) = two_sum(p, c.hi);
        quick_two_sum(s, t + (e + c.lo))
    }

    /// `self · b` for a plain `f32` `b`.
    #[inline(always)]
    pub fn mul_f32(self, b: f32) -> Self {
        let p = prod(self.hi, b);
        let e = self.hi.mul_add(b, -p);
        quick_two_sum(p, self.lo.mul_add(b, e))
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        if self.hi < 0.0 { -self } else { self }
    }

    /// `1 / √self` from the `f32` estimate `y`: with `r = 1 − self · y²` formed to full width by
    /// fused multiply-adds, `y (1 + r/2 + 3r²/8)` leaves an error of order `r³`, far below the
    /// format's.
    #[inline(always)]
    pub fn rsqrt(self) -> Self {
        let y = 1.0 / self.hi.sqrt();
        let p = prod(y, y);
        let pe = y.mul_add(y, -p);
        let r = (-self.hi).mul_add(p, 1.0);
        let r = (-self.hi).mul_add(pe, (-self.lo).mul_add(p, r));
        let c = prod(prod(y, r), r.mul_add(0.375, 0.5));
        quick_two_sum(y, c)
    }
}

impl Add for Ef64 {
    type Output = Self;
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        let (s, e) = two_sum(self.hi, b.hi);
        let (t, f) = two_sum(self.lo, b.lo);
        let r = quick_two_sum(s, e + t);
        quick_two_sum(r.hi, r.lo + f)
    }
}

impl Neg for Ef64 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self { hi: -self.hi, lo: -self.lo }
    }
}

impl Sub for Ef64 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}

impl Mul for Ef64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, b: Self) -> Self {
        let p = prod(self.hi, b.hi);
        let e = self.hi.mul_add(b.hi, -p);
        quick_two_sum(p, self.hi.mul_add(b.lo, self.lo.mul_add(b.hi, e)))
    }
}

impl Div for Ef64 {
    type Output = Self;
    #[inline(always)]
    /// The `f32` quotient, corrected by the remainder `self − b q₁`, whose leading part one
    /// fused multiply-add forms almost exactly (it is small, so its rounding is too).
    fn div(self, b: Self) -> Self {
        let q1 = self.hi / b.hi;
        let r = (-b.hi).mul_add(q1, self.hi) + self.lo;
        let r = (-b.lo).mul_add(q1, r);
        quick_two_sum(q1, r / b.hi)
    }
}

impl AddAssign for Ef64 {
    #[inline(always)]
    fn add_assign(&mut self, b: Self) {
        *self = *self + b;
    }
}

impl SubAssign for Ef64 {
    #[inline(always)]
    fn sub_assign(&mut self, b: Self) {
        *self = *self - b;
    }
}

impl MulAssign for Ef64 {
    #[inline(always)]
    fn mul_assign(&mut self, b: Self) {
        *self = *self * b;
    }
}

impl PartialOrd for Ef64 {
    #[inline(always)]
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        if self < b {
            Some(Ordering::Less)
        } else if self > b {
            Some(Ordering::Greater)
        } else if self == b {
            Some(Ordering::Equal)
        } else {
            None
        }
    }
    #[inline(always)]
    fn lt(&self, b: &Self) -> bool {
        self.hi < b.hi || (self.hi == b.hi && self.lo < b.lo)
    }
    #[inline(always)]
    fn le(&self, b: &Self) -> bool {
        self.hi < b.hi || (self.hi == b.hi && self.lo <= b.lo)
    }
    #[inline(always)]
    fn gt(&self, b: &Self) -> bool {
        b.lt(self)
    }
    #[inline(always)]
    fn ge(&self, b: &Self) -> bool {
        b.le(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ef(v: f64) -> Ef64 {
        let hi = v as f32;
        Ef64 { hi, lo: (v - hi as f64) as f32 }
    }

    fn val(e: Ef64) -> f64 {
        e.hi as f64 + e.lo as f64
    }

    fn close(got: Ef64, want: f64, ulps: f64) {
        let err = (val(got) - want).abs() / want.abs().max(f64::MIN_POSITIVE);
        assert!(err <= ulps * 2f64.powi(-48), "{} vs {want}: relative error {err:e}", val(got));
    }

    #[test]
    fn arithmetic_keeps_48_bits() {
        let xs = [1.0 / 3.0, -std::f64::consts::E, 1e-7 * std::f64::consts::PI, 12345.678901234567, 1.0 + 1e-12];
        for &a in &xs {
            for &b in &xs {
                close(ef(a) + ef(b), a + b, 4.0 * (a.abs() + b.abs()) / (a + b).abs());
                close(ef(a) * ef(b), a * b, 4.0);
                close(ef(a) / ef(b), a / b, 8.0);
                close(ef(a).mul_add(ef(b), ef(a)), a * b + a, 8.0 * (a * b).abs().max(a.abs()) / (a * b + a).abs());
            }
            if a > 0.0 {
                close(ef(a).rsqrt(), 1.0 / a.sqrt(), 8.0);
            }
        }
    }

    #[test]
    fn ordering_uses_the_low_word() {
        let a = ef(1.0 + 1e-12);
        let b = ef(1.0);
        assert!(a > b && b < a && a >= a && a.partial_cmp(&a) == Some(core::cmp::Ordering::Equal));
        assert!(-a < -b);
    }
}
