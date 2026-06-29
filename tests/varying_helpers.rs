use hydroplane::{Backend, Gang, Kernel, Scalar, SimdDispatch, dispatch, run_scalar};

struct Out<T> {
    map: Vec<T>,
    zip_map: Vec<T>,
    any: bool,
    all: bool,
    zip_any: bool,
    zip_all: bool,
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
            .to_f64();

        Out {
            map,
            zip_map,
            any,
            all,
            zip_any,
            zip_all,
            zip3,
            total: g.total(a).to_f64(),
            dot: g.dot(a, b).to_f64(),
            norm_sq: g.norm_sq(a).to_f64(),
            norm: g.norm(a).to_f64(),
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

        let map_want: Vec<f64> = a.iter().map(|&x| 2.0 * x.to_f64() + 1.0).collect();
        let zip_map_want: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| x.to_f64() * y.to_f64())
            .collect();
        let any_want = a.iter().any(|&x| x.to_f64() > 3.0);
        let all_want = a.iter().all(|&x| x.to_f64() < 3.0);
        let zip_any_want = a.iter().zip(&b).any(|(&x, &y)| x.to_f64() > y.to_f64());
        let zip_all_want = a.iter().zip(&b).all(|(&x, &y)| x.to_f64() <= y.to_f64());
        let zip3_want: f64 = a
            .iter()
            .zip(&b)
            .zip(&c)
            .map(|((&x, &y), &z)| x.to_f64() * y.to_f64() + z.to_f64())
            .sum();
        let total_want: f64 = a.iter().map(|&x| x.to_f64()).sum();
        let dot_want: f64 = a.iter().zip(&b).map(|(&x, &y)| x.to_f64() * y.to_f64()).sum();
        let norm_sq_want: f64 = a.iter().map(|&x| x.to_f64() * x.to_f64()).sum();
        let norm_want = norm_sq_want.sqrt();

        for (label, out) in [
            ("scalar", run_scalar(Helpers { a: &a, b: &b, c: &c })),
            ("dispatch", dispatch(Helpers { a: &a, b: &b, c: &c })),
        ] {
            assert_eq!(out.map.len(), map_want.len(), "{label} map len n={n}");
            for (i, &got) in out.map.iter().enumerate() {
                assert!(
                    close(got.to_f64(), map_want[i]),
                    "{label} map mismatch n={n} i={i}: got {}, want {}",
                    got.to_f64(),
                    map_want[i]
                );
            }
            assert_eq!(out.zip_map.len(), zip_map_want.len(), "{label} zip_map len n={n}");
            for (i, &got) in out.zip_map.iter().enumerate() {
                assert!(
                    close(got.to_f64(), zip_map_want[i]),
                    "{label} zip_map mismatch n={n} i={i}: got {}, want {}",
                    got.to_f64(),
                    zip_map_want[i]
                );
            }
            assert_eq!(out.any, any_want, "{label} any mismatch n={n}");
            assert_eq!(out.all, all_want, "{label} all mismatch n={n}");
            assert_eq!(out.zip_any, zip_any_want, "{label} zip_any mismatch n={n}");
            assert_eq!(out.zip_all, zip_all_want, "{label} zip_all mismatch n={n}");
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
