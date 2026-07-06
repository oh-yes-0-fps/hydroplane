//! Runtime cache of the unroll factor K: the count of independent FP accumulator chains that
//! saturates this core's FMA pipes, resolved once by [`Gang::detect_unroll`] and read by the
//! dispatch adapter to pick the `Unroll<_, K>` wrapper. `--cfg hp_no_ilp` (and SPIR-V) folds it to `1`.
//!
//! [`Gang::detect_unroll`]: crate::Gang::detect_unroll

#[cfg(not(any(hp_no_ilp, target_arch = "spirv")))]
mod imp {
    use core::sync::atomic::{AtomicU8, Ordering};

    static UNROLL: AtomicU8 = AtomicU8::new(0);

    /// `0` means unresolved; once resolved it is one of `{1,2,4,8,12,16}` and immutable for the
    /// life of the process.
    #[inline]
    pub(crate) fn cached() -> u8 {
        UNROLL.load(Ordering::Relaxed)
    }

    /// Idempotent: racing threads re-measure the same machine and store the same factor.
    #[inline]
    pub(crate) fn store(k: u8) {
        UNROLL.store(k, Ordering::Relaxed);
    }
}

#[cfg(any(hp_no_ilp, target_arch = "spirv"))]
mod imp {
    /// ILP compiled out: no atomic, no sweep, the factor is always `1`.
    #[inline(always)]
    pub(crate) fn cached() -> u8 {
        1
    }
}

pub(crate) use imp::cached;
#[cfg(not(any(hp_no_ilp, target_arch = "spirv")))]
pub(crate) use imp::store;
