//! Runtime unroll-factor (K) cache — the count of independent FP accumulator chains that saturates
//! this core's FMA pipes. Resolved once (a startup sweep, see [`Gang::detect_unroll`]), a relaxed
//! atomic load thereafter. See ../future/ILP_SUPERSCALAR.md.
//!
//! Building with `--cfg no_ilp` (and always on the SPIR-V target, whose lanes are the GPU's, not a
//! CPU's FMA pipes) compiles the whole multi-accumulator path out: this cache, the startup sweep,
//! and the `K`-unrolled reduction loops all collapse to the single-chain fold. `cached()` then folds
//! to the constant `1` so the rest of the crate keeps one code shape.
//!
//! [`Gang::detect_unroll`]: crate::Gang::detect_unroll

#[cfg(not(any(no_ilp, target_arch = "spirv")))]
mod imp {
    use core::sync::atomic::{AtomicU8, Ordering};

    static UNROLL: AtomicU8 = AtomicU8::new(0);

    /// `0` means unresolved; once resolved it is one of the candidate factors `{1,2,4,8,12,16}` and
    /// immutable for the life of the process.
    #[inline]
    pub(crate) fn cached() -> u8 {
        UNROLL.load(Ordering::Relaxed)
    }

    /// Idempotent: a racing thread that re-measures the same machine and stores the same factor again
    /// is harmless, since the saturation point is a property of the core, not of the call.
    #[inline]
    pub(crate) fn store(k: u8) {
        UNROLL.store(k, Ordering::Relaxed);
    }
}

#[cfg(any(no_ilp, target_arch = "spirv"))]
mod imp {
    /// ILP compiled out: no atomic, no sweep. The single resolved factor is `1`, so every reduction
    /// runs the one-chain fold and nothing ever needs to be stored.
    #[inline(always)]
    pub(crate) fn cached() -> u8 {
        1
    }
}

pub(crate) use imp::cached;
#[cfg(not(any(no_ilp, target_arch = "spirv")))]
pub(crate) use imp::store;
