//! Symmetric band matrices: chains over time or along a path (trajectory optimization, retiming
//! QPs with the constraints eliminated, 1D models), where every row couples only its few
//! neighbours. No ordering, symbolic analysis or index lookups: the factorization streams down the
//! band in `O(n·w²)` for half-bandwidth `w`.

use crate::cholesky::Error;
use crate::ldlt::Inertia;

/// The lower band of a symmetric `n × n` matrix with half-bandwidth `w`: `band[i·(w+1) + k] =
/// A[i][i−k]`, `k = 0` the diagonal (entries left of column 0 are ignored).
#[derive(Clone, Debug)]
pub struct Band {
    n: usize,
    w: usize,
    band: Vec<f64>,
}

impl Band {
    pub fn zeros(n: usize, w: usize) -> Self {
        Self { n, w, band: vec![0.0; n * (w + 1)] }
    }

    /// From the lower triangle in compressed sparse columns, its bandwidth found from the pattern.
    pub fn from_lower(col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Result<Self, Error> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        let mut w = 0;
        for j in 0..n {
            for &i in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
                if i < j || i >= n {
                    return Err(Error::Pattern);
                }
                w = w.max(i - j);
            }
        }
        let mut band = Self::zeros(n, w);
        for j in 0..n {
            for p in col_ptr[j]..col_ptr[j + 1] {
                *band.at_mut(row_idx[p], j) += values[p];
            }
        }
        Ok(band)
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    pub fn bandwidth(&self) -> usize {
        self.w
    }

    /// `A[i][j]` for `j ≤ i ≤ j + w`.
    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        assert!(j <= i && i - j <= self.w && i < self.n);
        &mut self.band[i * (self.w + 1) + (i - j)]
    }

    /// `y = A x`.
    pub fn multiply(&self, x: &[f64], y: &mut [f64]) {
        let (n, w) = (self.n, self.w);
        y.fill(0.0);
        for i in 0..n {
            let row = &self.band[i * (w + 1)..(i + 1) * (w + 1)];
            y[i] += row[0] * x[i];
            for k in 1..=w.min(i) {
                y[i] += row[k] * x[i - k];
                y[i - k] += row[k] * x[i];
            }
        }
    }
}

/// `A = L D Lᵀ` of a [`Band`], `L` unit lower with the same band, without pivoting: for a
/// positive definite band (then `D > 0`, a Cholesky factor in disguise) or a quasi-definite one
/// with expected pivot signs (a KKT chain), replaced by `±delta` when wrong or below `epsilon` as
/// [`crate::ldlt::Pivoting::Regularized`] does.
#[derive(Clone, Debug)]
pub struct BandedLdlt {
    factor: Band,
    regularized: usize,
}

impl BandedLdlt {
    /// Factors `a` taking every pivot as it comes; a zero pivot fails with [`Error::Singular`].
    pub fn new(a: Band) -> Result<Self, Error> {
        Self::factor(a, None)
    }

    /// Factors `a`, replacing a pivot whose sign disagrees with `signs` (or of magnitude below
    /// `1e-13`) by `±1e-9`: [`crate::ldlt::Pivoting::regularized`]'s rule.
    pub fn regularized(a: Band, signs: &[i8]) -> Result<Self, Error> {
        assert_eq!(signs.len(), a.n, "one expected sign per row");
        Self::factor(a, Some((signs, 1e-9, 1e-13)))
    }

    /// Factors a band that must be positive definite: [`Error::NotPositiveDefinite`] otherwise.
    pub fn positive_definite(a: Band) -> Result<Self, Error> {
        let f = Self::factor(a, None).map_err(|_| Error::NotPositiveDefinite)?;
        if f.inertia().positive == f.factor.n { Ok(f) } else { Err(Error::NotPositiveDefinite) }
    }

    fn factor(mut a: Band, regularize: Option<(&[i8], f64, f64)>) -> Result<Self, Error> {
        let (n, w) = (a.n, a.w);
        let stride = w + 1;
        let band = &mut a.band;
        let mut regularized = 0;
        let mut scaled = vec![0.0; w];
        for j in 0..n {
            let first = j.saturating_sub(w);
            for (s, k) in scaled.iter_mut().zip(first..j) {
                *s = band[j * stride + (j - k)] * band[k * stride];
            }
            let mut d = band[j * stride];
            for (s, k) in scaled.iter().zip(first..j) {
                d -= s * band[j * stride + (j - k)];
            }
            if let Some((signs, delta, epsilon)) = regularize {
                if signs[j] > 0 && d <= epsilon {
                    d = delta;
                    regularized += 1;
                } else if signs[j] < 0 && d >= -epsilon {
                    d = -delta;
                    regularized += 1;
                }
            }
            if d == 0.0 || !d.is_finite() {
                return Err(Error::Singular);
            }
            band[j * stride] = d;
            for i in j + 1..(j + w + 1).min(n) {
                let mut v = band[i * stride + (i - j)];
                for (s, k) in scaled.iter().zip(first..j) {
                    if i - k <= w {
                        v -= band[i * stride + (i - k)] * s;
                    }
                }
                band[i * stride + (i - j)] = v / d;
            }
        }
        Ok(Self { factor: a, regularized })
    }

    /// Pivots replaced by [`Self::regularized`].
    pub fn regularized_count(&self) -> usize {
        self.regularized
    }

    /// The signs of `D`: the eigenvalue signs of `A` (Sylvester).
    pub fn inertia(&self) -> Inertia {
        let stride = self.factor.w + 1;
        let mut inertia = Inertia::default();
        for i in 0..self.factor.n {
            let d = self.factor.band[i * stride];
            if d > 0.0 {
                inertia.positive += 1;
            } else if d < 0.0 {
                inertia.negative += 1;
            } else {
                inertia.zero += 1;
            }
        }
        inertia
    }

    /// `x ← A⁻¹ x`.
    pub fn solve_in_place(&self, x: &mut [f64]) {
        let (n, w) = (self.factor.n, self.factor.w);
        let stride = w + 1;
        let band = &self.factor.band;
        assert_eq!(x.len(), n);
        for i in 0..n {
            let mut v = x[i];
            for k in i.saturating_sub(w)..i {
                v -= band[i * stride + (i - k)] * x[k];
            }
            x[i] = v;
        }
        for i in 0..n {
            x[i] /= band[i * stride];
        }
        for i in (0..n).rev() {
            let mut v = x[i];
            for k in i + 1..(i + w + 1).min(n) {
                v -= band[k * stride + (k - i)] * x[k];
            }
            x[i] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(n: usize, w: usize, shift: f64) -> Band {
        let mut a = Band::zeros(n, w);
        for i in 0..n {
            *a.at_mut(i, i) = 2.0 * w as f64 + 1.0 + shift;
            for k in 1..=w.min(i) {
                *a.at_mut(i, i - k) = -1.0 / k as f64;
            }
        }
        a
    }

    fn check(a: &Band, f: &BandedLdlt) -> f64 {
        let b: Vec<f64> = (0..a.dim()).map(|i| ((i * 37) % 11) as f64 - 5.0).collect();
        let mut x = b.clone();
        f.solve_in_place(&mut x);
        let mut ax = vec![0.0; b.len()];
        a.multiply(&x, &mut ax);
        ax.iter().zip(&b).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
    }

    #[test]
    fn positive_definite_bands_solve() {
        for w in [1, 3, 8] {
            let a = chain(500, w, 0.1);
            let f = BandedLdlt::positive_definite(a.clone()).unwrap();
            assert!(check(&a, &f) < 1e-12, "w {w}");
        }
    }

    #[test]
    fn an_indefinite_band_reports_its_inertia() {
        let a = chain(300, 2, -5.4);
        let f = BandedLdlt::new(a.clone()).unwrap();
        let inertia = f.inertia();
        assert!(inertia.negative > 0 && inertia.positive > 0, "{inertia:?}");
        assert!(check(&a, &f) < 1e-10);
        assert!(BandedLdlt::positive_definite(a).is_err());
    }

    #[test]
    fn from_lower_finds_the_bandwidth() {
        let col_ptr = [0, 2, 4, 5];
        let row_idx = [0, 2, 1, 2, 2];
        let values = [4.0, 1.0, 3.0, 0.5, 2.0];
        let a = Band::from_lower(&col_ptr, &row_idx, &values).unwrap();
        assert_eq!(a.bandwidth(), 2);
        let f = BandedLdlt::positive_definite(a.clone()).unwrap();
        assert!(check(&a, &f) < 1e-14);
    }
}
