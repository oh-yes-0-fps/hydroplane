//! Fixed-arity bundles of SoA columns: [`Cols`] owns `N` parallel `Vec<T>` planes and hands out
//! `[&[T]; N]` / `[&mut [T]; N]` views. The planes are unpadded, so kernels over `Cols` views
//! use the masked-tail combinators (or `load_partial`) rather than sentinel padding.

use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use crate::scalar::Scalar;

/// `N` parallel `Vec<T>` planes of one logical table. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct Cols<T, const N: usize>(pub [Vec<T>; N]);

impl<T, const N: usize> Default for Cols<T, N> {
    fn default() -> Self {
        Self(core::array::from_fn(|_| Vec::new()))
    }
}

impl<T: Scalar, const N: usize> Cols<T, N> {
    /// `N` zero-filled planes of `len` elements each.
    pub fn zeros(len: usize) -> Self {
        Self(core::array::from_fn(|_| vec![T::ZERO; len]))
    }

    /// `N` planes of `len` elements, plane `c` filled by `f(c)`.
    pub fn filled(len: usize, mut f: impl FnMut(usize) -> T) -> Self {
        Self(core::array::from_fn(|c| vec![f(c); len]))
    }

    /// Rows in the table: the length of the shortest plane (they are normally all equal).
    pub fn len(&self) -> usize {
        self.0.iter().map(Vec::len).min().unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Shared views of all planes, in the `[&[T]; N]` shape the column combinators take.
    pub fn refs(&self) -> [&[T]; N] {
        core::array::from_fn(|c| self.0[c].as_slice())
    }

    /// Shared views of one row range of every plane.
    pub fn refs_range(&self, r: Range<usize>) -> [&[T]; N] {
        core::array::from_fn(|c| &self.0[c][r.clone()])
    }

    /// Mutable views of all planes.
    pub fn muts(&mut self) -> [&mut [T]; N] {
        self.0.each_mut().map(|v| v.as_mut_slice())
    }

    /// Mutable views of one row range of every plane.
    pub fn muts_range(&mut self, r: Range<usize>) -> [&mut [T]; N] {
        self.0.each_mut().map(|v| &mut v[r.clone()])
    }
}

impl<T, const N: usize> From<[Vec<T>; N]> for Cols<T, N> {
    fn from(planes: [Vec<T>; N]) -> Self {
        Self(planes)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::Cols;

    #[test]
    fn views_and_ranges() {
        let mut c: Cols<f32, 3> = Cols::filled(5, |p| p as f32);
        assert_eq!(c.len(), 5);
        assert_eq!(c.refs()[2], &[2.0; 5][..]);
        assert_eq!(c.refs_range(1..3)[1].len(), 2);
        c.muts()[0][4] = 9.0;
        c.muts_range(0..2)[2][0] = -1.0;
        assert_eq!(c.0[0][4], 9.0);
        assert_eq!(c.0[2][0], -1.0);
        assert!(Cols::<f32, 2>::zeros(0).is_empty());
    }
}
