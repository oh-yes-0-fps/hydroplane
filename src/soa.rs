//! Generic columnar (structure-of-arrays) storage with SIMD-friendly padding.
//!
//! Replaces the hand-rolled `Vec<f32>` + `pad(16)` + `NaN`-fill pattern found in SoA
//! collision code (e.g. `wreck`'s `SpheresSoA`) with one type generic over the scalar `T`
//! and the column count. Each column is padded to a multiple of [`MAX_LANES`] so any
//! backend (up to AVX-512's 16-wide f32) loads only full registers — no remainder path —
//! and the inactive tail lanes are filled with a caller-chosen value per column (e.g. a
//! radius of `NaN`, so distance comparisons on padding always fail).
//!
//! Requires the `alloc` feature.

use alloc::vec;
use alloc::vec::Vec;

use crate::scalar::Scalar;
pub use crate::MAX_LANES;

#[inline]
fn pad_to(n: usize) -> usize {
    n.div_ceil(MAX_LANES) * MAX_LANES
}

/// Columnar storage of `cols` channels of `T`, each padded to a multiple of [`MAX_LANES`].
///
/// Memory layout is `[col0; padded][col1; padded]…` in one allocation.
#[derive(Debug)]
pub struct Soa<T: Scalar> {
    cols: usize,
    len: usize,
    padded: usize,
    /// Per-column fill value for inactive (padding) lanes.
    pad_fill: Vec<T>,
    buf: Vec<T>,
}

impl<T: Scalar> Soa<T> {
    /// New SoA with `cols` columns, padding lanes filled with `T::ZERO`.
    pub fn new(cols: usize) -> Self {
        Self::with_pad_fills(&vec![T::ZERO; cols])
    }

    /// New SoA whose column `c` fills its padding lanes with `pad_fills[c]`.
    pub fn with_pad_fills(pad_fills: &[T]) -> Self {
        Self {
            cols: pad_fills.len(),
            len: 0,
            padded: 0,
            pad_fill: pad_fills.to_vec(),
            buf: Vec::new(),
        }
    }

    /// Build a padded SoA by copying existing equal-length column slices, one per channel.
    ///
    /// The bridge for borrowed columnar data — e.g. the per-field slices of a `soa-rs`
    /// `#[derive(Soars)]` struct (`Soa::from_columns(&[s.x(), s.y(), s.z(), s.r()], …)`).
    /// Column `c` fills its inactive padding lanes with `pad_fills[c]`. Every column slice
    /// must have the same length.
    pub fn from_columns(columns: &[&[T]], pad_fills: &[T]) -> Self {
        debug_assert_eq!(columns.len(), pad_fills.len(), "column / pad-fill arity mismatch");
        let len = columns.first().map_or(0, |c| c.len());
        let cols = columns.len();
        let padded = pad_to(len);
        let mut s = Self {
            cols,
            len,
            padded,
            pad_fill: pad_fills.to_vec(),
            buf: vec![T::ZERO; cols * padded],
        };
        for (c, col) in columns.iter().enumerate() {
            debug_assert_eq!(col.len(), len, "column {c} length mismatch");
            s.buf[c * padded..c * padded + len].copy_from_slice(col);
        }
        s.fill_padding(len);
        s
    }

    /// Reserve capacity for `rows` rows up front.
    pub fn with_capacity(pad_fills: &[T], rows: usize) -> Self {
        let cols = pad_fills.len();
        let padded = pad_to(rows);
        let mut s = Self {
            cols,
            len: 0,
            padded,
            pad_fill: pad_fills.to_vec(),
            buf: vec![T::ZERO; cols * padded],
        };
        s.fill_padding(0);
        s
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Padded length of each column (a multiple of [`MAX_LANES`]).
    #[inline]
    pub fn padded(&self) -> usize {
        self.padded
    }

    /// Padded slice of column `c` (length [`Soa::padded`]). `c < cols()` is the caller's contract,
    /// checked only under `debug_assertions`.
    #[inline]
    pub fn column(&self, c: usize) -> &[T] {
        debug_assert!(c < self.cols);
        let p = self.padded;
        unsafe { self.buf.get_unchecked(c * p..(c + 1) * p) }
    }

    /// Active prefix of column `c` (length [`len`](Soa::len), no padding) — the slice the kernels
    /// actually iterate, as opposed to the padded [`column`](Soa::column).
    #[inline]
    pub fn column_active(&self, c: usize) -> &[T] {
        debug_assert!(c < self.cols);
        let p = self.padded;
        unsafe { self.buf.get_unchecked(c * p..c * p + self.len) }
    }

    /// Mutable padded slice of column `c`. The caller must keep padding lanes consistent
    /// (`[len..padded]` are inactive) and pass `c < cols()` (checked only under `debug_assertions`);
    /// prefer [`Soa::push_row`] for appends.
    #[inline]
    pub fn column_mut(&mut self, c: usize) -> &mut [T] {
        debug_assert!(c < self.cols);
        let p = self.padded;
        unsafe { self.buf.get_unchecked_mut(c * p..(c + 1) * p) }
    }

    /// The active prefixes of all `N` columns, mutable and simultaneous — what an in-place transform
    /// (`(xs, ys, zs)` rotated together) needs and [`column_mut`](Soa::column_mut) can't give, since
    /// it borrows one column at a time. `N` must equal [`cols()`](Soa::cols).
    #[inline]
    pub fn columns_active_mut<const N: usize>(&mut self) -> [&mut [T]; N] {
        debug_assert_eq!(N, self.cols, "Soa::columns_active_mut: N must equal cols()");
        let p = self.padded;
        let len = self.len;
        let base = self.buf.as_mut_ptr();
        // Column `c` occupies `[c*p .. c*p + p)`; the active prefixes `[c*p .. c*p + len)` (with
        // `len <= p`) are therefore pairwise disjoint, so the raw slices alias no memory.
        core::array::from_fn(|c| unsafe { core::slice::from_raw_parts_mut(base.add(c * p), len) })
    }

    /// Per-column padding fill values (the `pad_fills` the SoA was built with).
    #[inline]
    pub fn pad_fills(&self) -> &[T] {
        &self.pad_fill
    }

    /// Copy row `i`'s value from every column into `out` (`out.len() == cols`) — the columnar
    /// row reconstruction (`out[c] = column(c)[i]`).
    pub fn copy_row(&self, i: usize, out: &mut [T]) {
        debug_assert!(i < self.len, "Soa::copy_row: index out of bounds");
        debug_assert_eq!(out.len(), self.cols, "Soa::copy_row: out arity mismatch");
        let p = self.padded;
        for (c, slot) in out.iter_mut().enumerate() {
            *slot = self.buf[c * p + i];
        }
    }

    /// Append every active row of `other`, re-padding once, and leave `other` empty. `other` must
    /// share this SoA's column count.
    pub fn append(&mut self, other: &mut Self) {
        self.extend_from(other);
        other.clear();
    }

    /// Append every active row of `other` (leaving it untouched), re-padding once. `other` must
    /// share this SoA's column count.
    pub fn extend_from(&mut self, other: &Self) {
        debug_assert_eq!(self.cols, other.cols, "Soa::extend_from: column count mismatch");
        if other.len == 0 {
            return;
        }
        let new_len = self.len + other.len;
        let new_p = pad_to(new_len);
        let mut buf = vec![T::ZERO; self.cols * new_p];
        let (sp, op, sl, ol) = (self.padded, other.padded, self.len, other.len);
        for c in 0..self.cols {
            let dst = c * new_p;
            buf[dst..dst + sl].copy_from_slice(&self.buf[c * sp..c * sp + sl]);
            buf[dst + sl..dst + sl + ol].copy_from_slice(&other.buf[c * op..c * op + ol]);
        }
        self.buf = buf;
        self.padded = new_p;
        self.len = new_len;
        self.fill_padding(new_len);
    }

    /// Append one row (`row.len() == cols`).
    pub fn push_row(&mut self, row: &[T]) {
        debug_assert_eq!(row.len(), self.cols, "row arity mismatch");
        if self.len == self.padded {
            self.grow();
        }
        let p = self.padded;
        let i = self.len;
        for (c, &val) in row.iter().enumerate() {
            self.buf[c * p + i] = val;
        }
        self.len += 1;
    }

    /// Reset to empty (keeps the allocation, re-arms padding fills).
    pub fn clear(&mut self) {
        self.len = 0;
        self.fill_padding(0);
    }

    fn grow(&mut self) {
        let old_p = self.padded;
        let new_p = if old_p == 0 { MAX_LANES } else { old_p + MAX_LANES };
        let mut buf = vec![T::ZERO; self.cols * new_p];
        for c in 0..self.cols {
            let src = &self.buf[c * old_p..c * old_p + self.len];
            buf[c * new_p..c * new_p + self.len].copy_from_slice(src);
        }
        self.buf = buf;
        self.padded = new_p;
        self.fill_padding(self.len);
    }

    /// Fill `[from..padded]` of every column with its pad value.
    fn fill_padding(&mut self, from: usize) {
        let p = self.padded;
        for c in 0..self.cols {
            let fill = self.pad_fill[c];
            for slot in &mut self.buf[c * p + from..(c + 1) * p] {
                *slot = fill;
            }
        }
    }
}

impl<T: Scalar> Clone for Soa<T> {
    fn clone(&self) -> Self {
        Self {
            cols: self.cols,
            len: self.len,
            padded: self.padded,
            pad_fill: self.pad_fill.clone(),
            buf: self.buf.clone(),
        }
    }

    /// Reuses `self`'s existing allocations when their capacity allows, rather than reallocating.
    fn clone_from(&mut self, source: &Self) {
        self.cols = source.cols;
        self.len = source.len;
        self.padded = source.padded;
        self.pad_fill.clone_from(&source.pad_fill);
        self.buf.clone_from(&source.buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_and_layout() {
        // two columns; column 1 pads with NaN
        let mut s = Soa::<f32>::with_pad_fills(&[0.0, f32::NAN]);
        for i in 0..5 {
            s.push_row(&[i as f32, (i as f32) + 0.5]);
        }
        assert_eq!(s.len(), 5);
        assert_eq!(s.padded(), MAX_LANES); // 5 -> 16
        assert_eq!(&s.column(0)[..5], &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&s.column(1)[..5], &[0.5, 1.5, 2.5, 3.5, 4.5]);
        // active col-0 padding is 0, col-1 padding is NaN
        assert_eq!(s.column(0)[5], 0.0);
        assert!(s.column(1)[5].is_nan());
    }

    #[test]
    fn from_columns_pads_and_copies() {
        let xs = [0.0f32, 1.0, 2.0];
        let rs = [0.5f32, 0.6, 0.7];
        let s = Soa::from_columns(&[&xs, &rs], &[0.0, f32::NAN]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.cols(), 2);
        assert_eq!(s.padded(), MAX_LANES);
        assert_eq!(&s.column(0)[..3], &xs);
        assert_eq!(&s.column(1)[..3], &rs);
        assert_eq!(s.column(0)[3], 0.0);
        assert!(s.column(1)[3].is_nan());
    }

    #[test]
    fn from_columns_empty() {
        let s = Soa::<f32>::from_columns(&[&[], &[]], &[0.0, f32::NAN]);
        assert_eq!(s.len(), 0);
        assert_eq!(s.padded(), 0);
        assert_eq!(s.cols(), 2);
    }

    #[test]
    fn grow_across_boundary() {
        let mut s = Soa::<f64>::new(1);
        for i in 0..20 {
            s.push_row(&[i as f64]);
        }
        assert_eq!(s.len(), 20);
        assert_eq!(s.padded(), 32);
        for i in 0..20 {
            assert_eq!(s.column(0)[i], i as f64);
        }
        for i in 20..32 {
            assert_eq!(s.column(0)[i], 0.0); // pad fill
        }
    }

    #[test]
    fn column_active_and_copy_row() {
        let mut s = Soa::<f32>::with_pad_fills(&[0.0, f32::NAN]);
        for i in 0..3 {
            s.push_row(&[i as f32, -(i as f32)]);
        }
        assert_eq!(s.column_active(0), &[0.0, 1.0, 2.0]);
        assert_eq!(s.column_active(1), &[0.0, -1.0, -2.0]);
        let mut row = [0.0f32; 2];
        s.copy_row(2, &mut row);
        assert_eq!(row, [2.0, -2.0]);
        assert_eq!(s.pad_fills()[0], 0.0);
        assert!(s.pad_fills()[1].is_nan());
    }

    #[test]
    fn columns_active_mut_transforms_in_place() {
        let mut s = Soa::<f32>::new(3);
        for i in 0..5 {
            s.push_row(&[i as f32, i as f32, i as f32]);
        }
        let [xs, ys, zs] = s.columns_active_mut::<3>();
        for slot in xs.iter_mut() {
            *slot += 1.0;
        }
        for slot in ys.iter_mut() {
            *slot *= 2.0;
        }
        for slot in zs.iter_mut() {
            *slot = -*slot;
        }
        assert_eq!(s.column_active(0), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(s.column_active(1), &[0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(s.column_active(2), &[0.0, -1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn append_repads_and_empties_other() {
        let mut a = Soa::<f32>::with_pad_fills(&[0.0, f32::NAN]);
        let mut b = Soa::<f32>::with_pad_fills(&[0.0, f32::NAN]);
        for i in 0..10 {
            a.push_row(&[i as f32, i as f32]);
        }
        for i in 0..10 {
            b.push_row(&[100.0 + i as f32, 100.0 + i as f32]);
        }
        a.append(&mut b);
        assert_eq!(a.len(), 20);
        assert_eq!(a.padded(), 32);
        assert_eq!(b.len(), 0);
        assert_eq!(a.column_active(0)[9], 9.0);
        assert_eq!(a.column_active(0)[10], 100.0);
        assert_eq!(a.column_active(0)[19], 109.0);
        assert!(a.column(1)[20].is_nan()); // padding re-armed after the merge
    }

    #[test]
    fn clone_from_reuses_and_matches() {
        let mut src = Soa::<f32>::new(2);
        for i in 0..7 {
            src.push_row(&[i as f32, i as f32 + 0.5]);
        }
        let mut dst = Soa::<f32>::new(2);
        dst.push_row(&[42.0, 42.0]);
        dst.clone_from(&src);
        assert_eq!(dst.len(), src.len());
        assert_eq!(dst.column_active(0), src.column_active(0));
        assert_eq!(dst.column_active(1), src.column_active(1));
    }
}
