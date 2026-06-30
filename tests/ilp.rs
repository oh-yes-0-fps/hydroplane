use hydroplane::{Backend, Gang, Kernel, Scalar, SimdDispatch, dispatch, run_scalar};

fn oracle_dot<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x.to_f64() * y.to_f64())
        .sum()
}

fn dot_zip_reduce<T: Scalar, S: Backend<T>>(g: Gang<T, S>, a: &[T], b: &[T]) -> f64 {
    g.zip_reduce(
        a,
        b,
        T::ZERO,
        T::ZERO,
        g.splat(T::ZERO),
        |acc, x, y| x.fma(y, acc),
        |p, q| p + q,
    )
    .reduce_sum()
    .to_f64()
}

fn sum_reduce<T: Scalar, S: Backend<T>>(g: Gang<T, S>, a: &[T]) -> f64 {
    g.reduce(a, T::ZERO, g.splat(T::ZERO), |acc, x| acc + x, |p, q| p + q)
        .reduce_sum()
        .to_f64()
}

// The unroll factor K is now chosen automatically (dispatch wraps the backend in `Unroll<S, K>`),
// so a kernel can't pin K from the outside; both `zip_reduce` and `reduce` exercise whichever K this
// core resolved. The size sweep below crosses the K*lanes() window and tail boundaries either way.
struct Variants<'a, T: Scalar> {
    a: &'a [T],
    b: &'a [T],
}
impl<T: Scalar> Kernel<T> for Variants<'_, T> {
    type Output = [f64; 2];
    fn run<S: Backend<T>>(self, g: Gang<T, S>) -> [f64; 2] {
        [dot_zip_reduce(g, self.a, self.b), sum_reduce(g, self.a)]
    }
}

const SIZES: &[usize] = &[0, 1, 3, 5, 8, 15, 16, 31, 64, 257];

fn check_all<T: Scalar + SimdDispatch>() {
    for &n in SIZES {
        let a: Vec<T> = (0..n)
            .map(|i| T::from_f64((i % 13) as f64 * 0.25 - 1.5))
            .collect();
        let b: Vec<T> = (0..n)
            .map(|i| T::from_f64((i % 7) as f64 * 0.5 - 1.0))
            .collect();

        let want = oracle_dot(&a, &b);
        let sum_want: f64 = a.iter().map(|&x| x.to_f64()).sum();
        let tol = 1e-3 * (1.0 + want.abs());
        let sum_tol = 1e-3 * (1.0 + sum_want.abs());

        for (label, outs) in [
            ("scalar", run_scalar(Variants { a: &a, b: &b })),
            ("dispatch", dispatch(Variants { a: &a, b: &b })),
        ] {
            assert!(
                (outs[0] - want).abs() <= tol,
                "{label} dot mismatch n={n}: got {}, want {want}",
                outs[0]
            );
            assert!(
                (outs[1] - sum_want).abs() <= sum_tol,
                "{label} sum mismatch n={n}: got {}, want {sum_want}",
                outs[1]
            );
        }
    }
}

#[test]
fn zip_reduce_matches_oracle_f32() {
    check_all::<f32>();
}

#[test]
fn zip_reduce_matches_oracle_f64() {
    check_all::<f64>();
}

struct DetectProbe;
impl Kernel<f32> for DetectProbe {
    type Output = f64;
    fn run<S: Backend<f32>>(self, g: Gang<f32, S>) -> f64 {
        let a: Vec<f32> = (0..257).map(|i| (i % 13) as f32 * 0.25 - 1.5).collect();
        dot_zip_reduce(g, &a, &a)
    }
}

#[test]
fn detection_resolves_and_is_stable() {
    let first = dispatch(DetectProbe);
    let second = dispatch(DetectProbe);
    assert_eq!(first.to_bits(), second.to_bits());
    let k = hydroplane::ilp_detected_for_test();
    assert!(
        matches!(k, 1 | 2 | 4 | 8 | 12 | 16),
        "cached K not a candidate factor: {k}"
    );
}
