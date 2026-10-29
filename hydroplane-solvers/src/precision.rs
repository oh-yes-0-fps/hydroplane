/// A working precision of the factorization or of an iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    F32,
    F64,
    /// `f64` emulated in `f32` pairs ([`crate::Ef64`]), for GPUs without `f64`.
    Ef64,
}

impl Precision {
    /// Spacing of the format's numbers relative to 1: what a residual is a multiple of.
    pub const fn epsilon(self) -> f64 {
        match self {
            Precision::F32 => f32::EPSILON as f64,
            Precision::F64 => f64::EPSILON,
            Precision::Ef64 => 1.0 / (1u64 << 48) as f64,
        }
    }

    pub const fn bytes(self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F64 | Precision::Ef64 => 8,
        }
    }
}
