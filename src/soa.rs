//! Columnar (SoA) storage padded to [`MAX_LANES`] so kernels never need a remainder path;
//! tail lanes take a caller-chosen fill (e.g. `NaN` radius). Requires `alloc`.

use alloc::vec;
use alloc::vec::Vec;

use crate::scalar::Scalar;
pub use crate::MAX_LANES;

#[inline]
fn pad_to(n: usize) -> usize {
    n.div_ceil(MAX_LANES) * MAX_LANES
}

/// Distance between column starts for columns of `cap` elements: `cap`, plus one cache line when
/// `cap` spans an even number of lines. Columns a power-of-two size apart all start in the same
/// L1 set, and a kernel streaming a dozen of them thrashes that set (a 12-in, 3-out map over
/// 4096-row `f32` columns ran 2.2× slower back to back than one line apart).
#[inline]
fn column_stride<T>(cap: usize) -> usize {
    const LINE: usize = 64;
    let size = size_of::<T>().max(1);
    if cap > 0 && LINE.is_multiple_of(size) && (cap * size).is_multiple_of(2 * LINE) { cap + LINE / size } else { cap }
}

/// Columnar storage of `cols` channels of `T`, each padded to a multiple of [`MAX_LANES`].
///
/// The columns share one allocation, spaced so consecutive columns start in different cache
/// sets; row capacity grows geometrically, so [`push_row`](Soa::push_row) is amortized `O(cols)`.
#[derive(Debug)]
pub struct Soa<T: Scalar> {
    cols: usize,
    len: usize,
    padded: usize,
    /// Rows each column has room for (a multiple of [`MAX_LANES`], at least `padded`).
    cap: usize,
    /// Elements from one column's start to the next's.
    stride: usize,
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
        Self { cols: pad_fills.len(), len: 0, padded: 0, cap: 0, stride: 0, pad_fill: pad_fills.to_vec(), buf: Vec::new() }
    }

    fn sized(pad_fills: &[T], padded: usize) -> Self {
        let cols = pad_fills.len();
        let stride = column_stride::<T>(padded);
        Self { cols, len: 0, padded, cap: padded, stride, pad_fill: pad_fills.to_vec(), buf: vec![T::ZERO; cols * stride] }
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
        let mut s = Self::sized(pad_fills, pad_to(len));
        s.len = len;
        for (c, col) in columns.iter().enumerate() {
            debug_assert_eq!(col.len(), len, "column {c} length mismatch");
            let at = c * s.stride;
            s.buf[at..at + len].copy_from_slice(col);
        }
        s.fill_padding(len);
        s
    }

    /// Reserve capacity for `rows` rows up front.
    pub fn with_capacity(pad_fills: &[T], rows: usize) -> Self {
        let mut s = Self::sized(pad_fills, pad_to(rows));
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

    /// Padded slice of column `c` (length [`Soa::padded`]). Panics if `c >= cols()`.
    #[inline]
    pub fn column(&self, c: usize) -> &[T] {
        assert!(c < self.cols);
        let at = c * self.stride;
        // SAFETY: `c < cols` and `padded <= cap <= stride`, inside `buf` (`cols * stride` long).
        unsafe { self.buf.get_unchecked(at..at + self.padded) }
    }

    /// Active prefix of column `c` (length [`len`](Soa::len), no padding), as opposed to the
    /// padded [`column`](Soa::column).
    #[inline]
    pub fn column_active(&self, c: usize) -> &[T] {
        assert!(c < self.cols);
        let at = c * self.stride;
        // SAFETY: as in `column`, with `len <= padded`.
        unsafe { self.buf.get_unchecked(at..at + self.len) }
    }

    /// Mutable padded slice of column `c`; panics if `c >= cols()`. The caller must keep padding
    /// lanes consistent (`[len..padded]` are inactive); prefer [`Soa::push_row`] for appends.
    #[inline]
    pub fn column_mut(&mut self, c: usize) -> &mut [T] {
        assert!(c < self.cols);
        let at = c * self.stride;
        // SAFETY: as in `column`.
        unsafe { self.buf.get_unchecked_mut(at..at + self.padded) }
    }

    /// The active prefixes of all `N` columns, mutable and simultaneous — what an in-place
    /// transform (`(xs, ys, zs)` rotated together) needs and [`column_mut`](Soa::column_mut),
    /// borrowing one column at a time, can't give. `N` must equal [`cols()`](Soa::cols).
    #[inline]
    pub fn columns_active_mut<const N: usize>(&mut self) -> [&mut [T]; N] {
        assert_eq!(N, self.cols, "Soa::columns_active_mut: N must equal cols()");
        let (stride, len) = (self.stride, self.len);
        let base = self.buf.as_mut_ptr();
        // SAFETY: the active prefixes `[c*stride .. c*stride + len)` with `len <= stride` are
        // pairwise disjoint and inside `buf`.
        core::array::from_fn(|c| unsafe { core::slice::from_raw_parts_mut(base.add(c * stride), len) })
    }

    /// Per-column padding fill values (the `pad_fills` the SoA was built with).
    #[inline]
    pub fn pad_fills(&self) -> &[T] {
        &self.pad_fill
    }

    /// Copy row `i`'s value from every column into `out` (`out.len() == cols`):
    /// `out[c] = column(c)[i]`.
    pub fn copy_row(&self, i: usize, out: &mut [T]) {
        debug_assert!(i < self.len, "Soa::copy_row: index out of bounds");
        debug_assert_eq!(out.len(), self.cols, "Soa::copy_row: out arity mismatch");
        for (c, slot) in out.iter_mut().enumerate() {
            *slot = self.buf[c * self.stride + i];
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
        self.reserve(new_p);
        for c in 0..self.cols {
            let dst = c * self.stride + self.len;
            let src = c * other.stride;
            self.buf[dst..dst + other.len].copy_from_slice(&other.buf[src..src + other.len]);
        }
        self.padded = new_p;
        self.len = new_len;
        self.fill_padding(new_len);
    }

    /// Append one row (`row.len() == cols`).
    pub fn push_row(&mut self, row: &[T]) {
        debug_assert_eq!(row.len(), self.cols, "row arity mismatch");
        if self.len == self.padded {
            self.reserve(self.padded + MAX_LANES);
            self.padded += MAX_LANES;
            self.fill_padding(self.len);
        }
        let i = self.len;
        for (c, &val) in row.iter().enumerate() {
            self.buf[c * self.stride + i] = val;
        }
        self.len += 1;
    }

    /// Reset to empty (keeps the allocation, re-arms padding fills).
    pub fn clear(&mut self) {
        self.len = 0;
        self.fill_padding(0);
    }

    /// Room for `rows` rows per column, at least doubling when it has to move.
    fn reserve(&mut self, rows: usize) {
        if rows <= self.cap {
            return;
        }
        let cap = rows.max(2 * self.cap);
        let stride = column_stride::<T>(cap);
        let mut buf = vec![T::ZERO; self.cols * stride];
        for c in 0..self.cols {
            buf[c * stride..c * stride + self.len].copy_from_slice(&self.buf[c * self.stride..c * self.stride + self.len]);
        }
        self.buf = buf;
        self.cap = cap;
        self.stride = stride;
    }

    /// Fill `[from..padded]` of every column with its pad value.
    fn fill_padding(&mut self, from: usize) {
        for c in 0..self.cols {
            let fill = self.pad_fill[c];
            let at = c * self.stride;
            self.buf[at + from..at + self.padded].fill(fill);
        }
    }
}

impl<T: Scalar> Clone for Soa<T> {
    fn clone(&self) -> Self {
        Self {
            cols: self.cols,
            len: self.len,
            padded: self.padded,
            cap: self.cap,
            stride: self.stride,
            pad_fill: self.pad_fill.clone(),
            buf: self.buf.clone(),
        }
    }

    /// Reuses `self`'s existing allocations when their capacity allows, rather than reallocating.
    fn clone_from(&mut self, source: &Self) {
        self.cols = source.cols;
        self.len = source.len;
        self.padded = source.padded;
        self.cap = source.cap;
        self.stride = source.stride;
        self.pad_fill.clone_from(&source.pad_fill);
        self.buf.clone_from(&source.buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_and_layout() {
        let mut s = Soa::<f32>::with_pad_fills(&[0.0, f32::NAN]);
        for i in 0..5 {
            s.push_row(&[i as f32, (i as f32) + 0.5]);
        }
        assert_eq!(s.len(), 5);
        assert_eq!(s.padded(), MAX_LANES);
        assert_eq!(&s.column(0)[..5], &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&s.column(1)[..5], &[0.5, 1.5, 2.5, 3.5, 4.5]);
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
            assert_eq!(s.column(0)[i], 0.0);
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
    fn power_of_two_columns_start_in_different_cache_sets() {
        let cols: [Vec<f32>; 4] = core::array::from_fn(|c| vec![c as f32; 4096]);
        let s = Soa::from_columns(&[&cols[0], &cols[1], &cols[2], &cols[3]], &[0.0; 4]);
        let set = |c: usize| (s.column(c).as_ptr() as usize / 64) % 64;
        assert!((1..4).all(|c| set(c) != set(0)));
        for (c, col) in cols.iter().enumerate() {
            assert_eq!(s.column_active(c), &col[..]);
        }
    }

    #[test]
    fn many_pushes_keep_every_row_and_fill() {
        let mut s = Soa::<f64>::with_pad_fills(&[-1.0, -2.0, -3.0]);
        for i in 0..5000 {
            s.push_row(&[i as f64, 2.0 * i as f64, 3.0 * i as f64]);
        }
        assert_eq!((s.len(), s.padded()), (5000, 5024));
        for c in 0..3 {
            let col = s.column(c);
            assert!((0..5000).all(|i| col[i] == (c + 1) as f64 * i as f64));
            assert!(col[5000..].iter().all(|&v| v == -((c + 1) as f64)));
        }
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
