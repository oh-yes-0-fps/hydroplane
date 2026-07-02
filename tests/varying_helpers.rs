use hydroplane::{Backend, Gang, Kernel, Scalar, SimdDispatch, dispatch, run_scalar};

struct Out<T> {
    map: Vec<T>,
    zip_map: Vec<T>,
    any: bool,
    all: bool,
    zip_any: bool,
    zip_all: bool,
    any_n: bool,
    all_n: bool,
    zip3: f64,
    total: f64,
    dot: f64,
    norm_sq: f64,
    norm: f64,
}

struct Helpers<'a, T: Scalar> {
    a: &'a [T],
    b: &'a [T],
    c: &'a [T],
}

impl<T: Scalar> Kernel<T> for Helpers<'_, T> {
    type Output = Out<T>;
    fn run<S: Backend<T>>(self, g: Gang<T, S>) -> Out<T> {
        let (a, b, c) = (self.a, self.b, self.c);

        let mut map = vec![T::ZERO; a.len()];
        g.map(a, &mut map, T::ZERO, |x| {
            x * T::from_f64(2.0) + T::from_f64(1.0)
        });

        let mut zip_map = vec![T::ZERO; a.len().min(b.len())];
        g.zip_map(a, b, &mut zip_map, T::ZERO, T::ZERO, |x, y| x * y);

        let three = T::from_f64(3.0);
        let any = g.any(a, T::from_f64(f64::NEG_INFINITY), |x| x.gt(g.splat(three)));
        let all = g.all(a, T::ZERO, |x| x.lt(g.splat(three)));

        let zip_any = g.zip_any(a, b, T::from_f64(f64::NEG_INFINITY), T::from_f64(f64::INFINITY), |x, y| {
            x.gt(y)
        });
        let zip_all = g.zip_all(a, b, T::from_f64(f64::NEG_INFINITY), T::from_f64(f64::INFINITY), |x, y| {
            x.le(y)
        });

        // N-column (3) any/all — active-masked tail, no sentinel fills.
        let ten = g.splat(T::from_f64(10.0));
        let any_n = g.any_n([a, b, c], |[x, y, z]| (x + y + z).gt(g.splat(three)));
        let all_n = g.all_n([a, b, c], |[x, y, z]| (x + y + z).lt(ten));

        let zip3 = g
            .zip3_fold(
                a,
                b,
                c,
                T::ZERO,
                T::ZERO,
                T::ZERO,
                g.splat(T::ZERO),
                |acc, x, y, z| acc + x * y + z,
            )
            .reduce_sum()
            .into_f64();

        Out {
            map,
            zip_map,
            any,
            all,
            zip_any,
            zip_all,
            any_n,
            all_n,
            zip3,
            total: g.total(a).into_f64(),
            dot: g.dot(a, b).into_f64(),
            norm_sq: g.norm_sq(a).into_f64(),
            norm: g.norm(a).into_f64(),
        }
    }
}

const SIZES: &[usize] = &[0, 1, 3, 5, 8, 15, 16, 31, 64, 257];

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-3 * (1.0 + want.abs())
}

fn check_all<T: Scalar + SimdDispatch>() {
    for &n in SIZES {
        let a: Vec<T> = (0..n)
            .map(|i| T::from_f64((i % 13) as f64 * 0.5 - 2.0))
            .collect();
        let b: Vec<T> = (0..n)
            .map(|i| T::from_f64((i % 7) as f64 * 0.5 - 1.0))
            .collect();
        let c: Vec<T> = (0..n)
            .map(|i| T::from_f64((i % 5) as f64 * 0.25 - 0.5))
            .collect();

        let map_want: Vec<f64> = a.iter().map(|&x| 2.0 * x.into_f64() + 1.0).collect();
        let zip_map_want: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| x.into_f64() * y.into_f64())
            .collect();
        let any_want = a.iter().any(|&x| x.into_f64() > 3.0);
        let all_want = a.iter().all(|&x| x.into_f64() < 3.0);
        let zip_any_want = a.iter().zip(&b).any(|(&x, &y)| x.into_f64() > y.into_f64());
        let zip_all_want = a.iter().zip(&b).all(|(&x, &y)| x.into_f64() <= y.into_f64());
        let any_n_want = (0..n).any(|i| a[i].into_f64() + b[i].into_f64() + c[i].into_f64() > 3.0);
        let all_n_want = (0..n).all(|i| a[i].into_f64() + b[i].into_f64() + c[i].into_f64() < 10.0);
        let zip3_want: f64 = a
            .iter()
            .zip(&b)
            .zip(&c)
            .map(|((&x, &y), &z)| x.into_f64() * y.into_f64() + z.into_f64())
            .sum();
        let total_want: f64 = a.iter().map(|&x| x.into_f64()).sum();
        let dot_want: f64 = a.iter().zip(&b).map(|(&x, &y)| x.into_f64() * y.into_f64()).sum();
        let norm_sq_want: f64 = a.iter().map(|&x| x.into_f64() * x.into_f64()).sum();
        let norm_want = norm_sq_want.sqrt();

        for (label, out) in [
            ("scalar", run_scalar(Helpers { a: &a, b: &b, c: &c })),
            ("dispatch", dispatch(Helpers { a: &a, b: &b, c: &c })),
        ] {
            assert_eq!(out.map.len(), map_want.len(), "{label} map len n={n}");
            for (i, &got) in out.map.iter().enumerate() {
                assert!(
                    close(got.into_f64(), map_want[i]),
                    "{label} map mismatch n={n} i={i}: got {}, want {}",
                    got.into_f64(),
                    map_want[i]
                );
            }
            assert_eq!(out.zip_map.len(), zip_map_want.len(), "{label} zip_map len n={n}");
            for (i, &got) in out.zip_map.iter().enumerate() {
                assert!(
                    close(got.into_f64(), zip_map_want[i]),
                    "{label} zip_map mismatch n={n} i={i}: got {}, want {}",
                    got.into_f64(),
                    zip_map_want[i]
                );
            }
            assert_eq!(out.any, any_want, "{label} any mismatch n={n}");
            assert_eq!(out.all, all_want, "{label} all mismatch n={n}");
            assert_eq!(out.zip_any, zip_any_want, "{label} zip_any mismatch n={n}");
            assert_eq!(out.zip_all, zip_all_want, "{label} zip_all mismatch n={n}");
            assert_eq!(out.any_n, any_n_want, "{label} any_n mismatch n={n}");
            assert_eq!(out.all_n, all_n_want, "{label} all_n mismatch n={n}");
            assert!(
                close(out.zip3, zip3_want),
                "{label} zip3_fold mismatch n={n}: got {}, want {zip3_want}",
                out.zip3
            );
            assert!(
                close(out.total, total_want),
                "{label} total mismatch n={n}: got {}, want {total_want}",
                out.total
            );
            assert!(
                close(out.dot, dot_want),
                "{label} dot mismatch n={n}: got {}, want {dot_want}",
                out.dot
            );
            assert!(
                close(out.norm_sq, norm_sq_want),
                "{label} norm_sq mismatch n={n}: got {}, want {norm_sq_want}",
                out.norm_sq
            );
            assert!(
                close(out.norm, norm_want),
                "{label} norm mismatch n={n}: got {}, want {norm_want}",
                out.norm
            );
        }
    }
}

#[test]
fn helpers_match_oracle_f32() {
    check_all::<f32>();
}

#[test]
fn helpers_match_oracle_f64() {
    check_all::<f64>();
}

struct MaskAbs<'a, T: Scalar> {
    a: &'a [T],
    cnt: usize,
}

impl<T: Scalar> Kernel<T> for MaskAbs<'_, T> {
    type Output = (Vec<T>, usize, Vec<T>);
    fn run<S: Backend<T>>(self, g: Gang<T, S>) -> (Vec<T>, usize, Vec<T>) {
        let lanes = g.lanes();
        let active = self.cnt.min(lanes);
        let m = g.active_mask(active);
        let mut flags = vec![T::ZERO; lanes];
        g.splat(T::ONE).select(m, g.splat(T::ZERO)).store(&mut flags);

        let mut absv = vec![T::ZERO; self.a.len()];
        g.map(self.a, &mut absv, T::ZERO, |x| x.abs());
        (flags, active, absv)
    }
}

fn check_mask_abs<T: Scalar + SimdDispatch>() {
    for &cnt in &[0usize, 1, 2, 5, 8, 100] {
        for &n in SIZES {
            let a: Vec<T> = (0..n)
                .map(|i| T::from_f64((i % 11) as f64 * 0.5 - 2.5))
                .collect();
            for (label, (flags, active, absv)) in [
                ("scalar", run_scalar(MaskAbs { a: &a, cnt })),
                ("dispatch", dispatch(MaskAbs { a: &a, cnt })),
            ] {
                for (i, f) in flags.iter().enumerate() {
                    let want = if i < active { 1.0 } else { 0.0 };
                    assert_eq!(f.into_f64(), want, "{label} active_mask cnt={cnt} lane={i}");
                }
                assert_eq!(absv.len(), a.len());
                for (i, got) in absv.iter().enumerate() {
                    assert_eq!(got.into_f64(), a[i].into_f64().abs(), "{label} abs n={n} i={i}");
                }
            }
        }
    }
}

#[test]
fn mask_abs_match_oracle_f32() {
    check_mask_abs::<f32>();
}

#[test]
fn mask_abs_match_oracle_f64() {
    check_mask_abs::<f64>();
}

/// `chunks_exact` must tile exactly the full-register prefix in order, and `remainder`
/// (both the `Gang` method and the iterator's) must describe precisely the leftover tail.
struct ExactCovers {
    len: usize,
}

impl Kernel<f32> for ExactCovers {
    type Output = bool;
    fn run<S: Backend<f32>>(self, g: Gang<f32, S>) -> bool {
        let n = g.lanes();
        let len = self.len;
        let it = g.chunks_exact(len);
        if it.remainder() != g.remainder(len) {
            return false;
        }
        let mut expect = 0usize;
        for off in it {
            if off != expect {
                return false;
            }
            expect += n;
        }
        match g.remainder(len) {
            Some((off, cnt)) => off == expect && cnt > 0 && cnt < n && off + cnt == len,
            None => expect == len,
        }
    }
}

struct SumExact<'a> {
    xs: &'a [f32],
}

impl Kernel<f32> for SumExact<'_> {
    type Output = f32;
    fn run<S: Backend<f32>>(self, g: Gang<f32, S>) -> f32 {
        let n = g.lanes();
        let mut acc = g.splat(0.0);
        for off in g.chunks_exact(self.xs.len()) {
            acc = acc + g.load(&self.xs[off..off + n]);
        }
        if let Some((off, cnt)) = g.remainder(self.xs.len()) {
            acc = acc + g.load_partial(&self.xs[off..off + cnt], 0.0);
        }
        acc.reduce_sum()
    }
}

#[test]
fn chunks_exact_and_remainder() {
    for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1000, 1003] {
        assert!(dispatch(ExactCovers { len }), "decomposition wrong at len={len}");
        assert!(run_scalar(ExactCovers { len }), "scalar decomposition wrong at len={len}");

        let xs: Vec<f32> = (0..len).map(|i| (i as f32 % 9.0) - 4.0).collect();
        let want: f32 = xs.iter().sum();
        let got = dispatch(SumExact { xs: &xs });
        assert!((got - want).abs() <= 1e-3 * (1.0 + want.abs()), "sum wrong at len={len}: {got} vs {want}");
    }
}
