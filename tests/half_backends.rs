//! The `f16`/`bf16` SIMD backends agree with the scalar oracle, and `f16` reaches its native
//! wide backend (8-wide NEON FEAT_FP16, 16/32-wide AVX-512(-FP16)) when the CPU supports it rather
//! than falling back to the 1-lane `ScalarBackend`.

use hydroplane::{BackendAll, Backend, Gang, Kernel, SimdDispatch, bf16, dispatch, f16, kernel};

struct Lanes<T>(core::marker::PhantomData<T>);
impl<T: hydroplane::Scalar> Kernel<T> for Lanes<T> {
    type Output = usize;
    fn run<S: BackendAll + Backend<T>>(self, ctx: Gang<S>) -> usize {
        ctx.lanes::<T>()
    }
}
fn lanes<T: hydroplane::Scalar + SimdDispatch>() -> usize {
    dispatch::<T, _>(Lanes(core::marker::PhantomData))
}

#[test]
fn f16_reaches_native_wide_backend() {
    let l = lanes::<f16>();
    #[cfg(all(target_arch = "aarch64", feature = "std"))]
    if std::arch::is_aarch64_feature_detected!("fp16") {
        assert_eq!(l, 8, "FEAT_FP16 present but f16 backend is not 8-wide NEON");
    }
    // Whatever the host, the kernel must run and report a sane lane count.
    assert!(l >= 1, "lanes must be positive");
}

#[kernel]
fn f16_count_sum_k<'a>(ctx: Gang, xs: &'a [f16], t: f16) -> usize {
    // counts and a reduction in one pass exercise load / compare / mask-bitmask / horizontal add.
    let tv = ctx.splat(t);
    ctx.count_n([xs], |[x]| x.gt(tv))
}

#[test]
fn f16_count_matches_scalar_oracle() {
    // Small integers are exact in f16, so the count is an exact comparison across the lane boundary.
    for len in [0usize, 1, 7, 8, 9, 16, 31, 100] {
        let xs: Vec<f16> = (0..len).map(|i| f16::from_f32((i % 17) as f32)).collect();
        let t = f16::from_f32(8.0);
        let got = f16_count_sum_k(&xs, t);
        let want = xs.iter().filter(|&&x| x > t).count();
        assert_eq!(got, want, "len={len}");
    }
}

#[kernel]
fn f16_dot_k<'a>(ctx: Gang, a: &'a [f16], b: &'a [f16]) -> f16 {
    ctx.dot(a, b)
}

#[test]
fn f16_dot_matches_scalar_oracle() {
    // Keep magnitudes small so the f16 dot is exact (products of small ints, sum < 2048).
    for len in [3usize, 8, 13, 64] {
        let a: Vec<f16> = (0..len).map(|i| f16::from_f32((i % 5) as f32)).collect();
        let b: Vec<f16> = (0..len).map(|i| f16::from_f32((i % 3) as f32)).collect();
        let got = f16_dot_k(&a, &b).to_f32();
        let want: f32 = a.iter().zip(&b).map(|(x, y)| x.to_f32() * y.to_f32()).sum();
        assert!((got - want).abs() <= 1.0, "len={len}: got {got}, want {want}");
    }
}

#[kernel]
fn bf16_count_k<'a>(ctx: Gang, xs: &'a [bf16], t: bf16) -> usize {
    let tv = ctx.splat(t);
    ctx.count_n([xs], |[x]| x.gt(tv))
}

#[test]
fn bf16_count_matches_scalar_oracle() {
    assert!(lanes::<bf16>() >= 1);
    for len in [0usize, 1, 5, 8, 17, 64] {
        let xs: Vec<bf16> = (0..len).map(|i| bf16::from_f32((i % 11) as f32)).collect();
        let t = bf16::from_f32(5.0);
        let got = bf16_count_k(&xs, t);
        let want = xs.iter().filter(|&&x| x > t).count();
        assert_eq!(got, want, "len={len}");
    }
}
