//! The host side of the kernels' emulated `f64`.

/// `f64` emulated as an unevaluated sum `hi + lo` of two `f32`s (about 48 significant bits), the
/// working precision of the GPU kernels on devices without `f64`. Laid out as the kernels' own
/// type, so a buffer of these is what they read and write.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ef64 {
    pub hi: f32,
    pub lo: f32,
}

impl Ef64 {
    /// The nearest pair: `hi` rounded from `v`, `lo` from what that left.
    pub fn from_f64(v: f64) -> Self {
        let hi = v as f32;
        Self { hi, lo: (v - hi as f64) as f32 }
    }

    pub fn to_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }
}

#[cfg(test)]
mod tests {
    use super::Ef64;

    #[test]
    fn round_trips_to_48_bits() {
        for v in [1.0 / 3.0, -std::f64::consts::E * 1e-9, 12345.678901234567, 1e30 / 7.0] {
            let e = Ef64::from_f64(v);
            assert!(((e.to_f64() - v) / v).abs() < 2f64.powi(-48), "{v}");
            assert!(e.lo.abs() <= e.hi.abs() * f32::EPSILON);
        }
    }
}
