//! Working buffers for the dense routines' packing and copies, reused per thread under `std`.
//!
//! A fresh buffer of a few hundred kilobytes comes from `mmap` and faults on every page as it is
//! first written; a 256² `f64` GEMM spent a third of its time there.

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

use crate::scalar::FloatScalar;

/// Buffers past this size go back to the allocator rather than staying with the thread.
#[cfg(feature = "std")]
const KEEP_BYTES: usize = 32 << 20;
#[cfg(feature = "std")]
const KEEP_COUNT: usize = 4;

#[cfg(feature = "std")]
std::thread_local! {
    static POOL: core::cell::RefCell<Vec<Vec<u64>>> = const { core::cell::RefCell::new(Vec::new()) };
}

/// `len` elements of `T` whose contents are unspecified (whatever an earlier use left there):
/// every reader must write before it reads.
pub(crate) struct Scratch<T> {
    words: Vec<u64>,
    len: usize,
    _t: PhantomData<T>,
}

impl<T: FloatScalar> Scratch<T> {
    pub(crate) fn new(len: usize) -> Self {
        const { assert!(align_of::<T>() <= align_of::<u64>()) };
        let mut words = take();
        let need = (len * size_of::<T>()).div_ceil(size_of::<u64>());
        if words.len() < need {
            words.resize(need, 0);
        }
        Scratch { words, len, _t: PhantomData }
    }
}

impl<T> Deref for Scratch<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `words` holds at least `len` `T`s, is aligned for `T` (checked in `new`), and every
        // bit pattern is a valid float.
        unsafe { core::slice::from_raw_parts(self.words.as_ptr().cast::<T>(), self.len) }
    }
}

impl<T> DerefMut for Scratch<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`, and `self` is borrowed mutably.
        unsafe { core::slice::from_raw_parts_mut(self.words.as_mut_ptr().cast::<T>(), self.len) }
    }
}

impl<T> Drop for Scratch<T> {
    fn drop(&mut self) {
        give(core::mem::take(&mut self.words));
    }
}

#[cfg(feature = "std")]
fn take() -> Vec<u64> {
    POOL.try_with(|p| p.borrow_mut().pop()).ok().flatten().unwrap_or_default()
}

#[cfg(feature = "std")]
fn give(words: Vec<u64>) {
    if words.capacity() * size_of::<u64>() <= KEEP_BYTES {
        let _ = POOL.try_with(|p| {
            let mut p = p.borrow_mut();
            if p.len() < KEEP_COUNT {
                p.push(words);
            }
        });
    }
}

#[cfg(not(feature = "std"))]
fn take() -> Vec<u64> {
    Vec::new()
}

#[cfg(not(feature = "std"))]
fn give(_: Vec<u64>) {}
