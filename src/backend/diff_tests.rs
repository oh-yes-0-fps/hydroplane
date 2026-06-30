//! Differential tests: every SIMD backend must agree with the [`ScalarBackend`] oracle,
//! op for op, lane for lane. These live in-crate because the SIMD tokens are `pub(crate)`
//! — application code reaches them only through `dispatch`, but the tests need to pin a
//! specific backend to verify it in isolation.
//!
//! Exact agreement is required for add/sub/mul/div/min/max/neg and the comparison/select
//! paths; sqrt/fma and the horizontal sum use a relative tolerance (fma may be fused;
//! reduce order differs).

use rand::Rng;

use super::{Backend, ScalarBackend};
use crate::scalar::Scalar;

type BinOp<T> = fn(ScalarBackend, T, T) -> T;

fn rel_eq<T: Scalar>(got: T, want: T, tol: f64) -> bool {
    let (g, w) = (got.to_f64(), want.to_f64());
    (g - w).abs() <= tol * (1.0 + w.abs())
}

fn check<T, S>(b: S, sample: impl Fn(&mut dyn FnMut() -> f64) -> T)
where
    T: Scalar,
    S: Backend<T>,
{
    let s = ScalarBackend;
    let n = b.lanes();
    let mut rng = rand::rng();
    let mut draw = || rng.random_range(-10.0f64..10.0);

    let store = |b: &S, v: S::Vector| {
        let mut o = vec![T::ZERO; n];
        b.store(v, &mut o);
        o
    };

    for _ in 0..2000 {
        let a: Vec<T> = (0..n).map(|_| sample(&mut draw)).collect();
        let bb: Vec<T> = (0..n).map(|_| sample(&mut draw)).collect();
        let va = b.load(&a);
        let vb = b.load(&bb);

        let exact: [(&str, Vec<T>, BinOp<T>); 6] = [
            ("add", store(&b, b.add(va, vb)), |s, x, y| s.add(x, y)),
            ("sub", store(&b, b.sub(va, vb)), |s, x, y| s.sub(x, y)),
            ("mul", store(&b, b.mul(va, vb)), |s, x, y| s.mul(x, y)),
            ("div", store(&b, b.div(va, vb)), |s, x, y| s.div(x, y)),
            ("min", store(&b, b.min(va, vb)), |s, x, y| s.min(x, y)),
            ("max", store(&b, b.max(va, vb)), |s, x, y| s.max(x, y)),
        ];
        for (name, got, op) in exact {
            for i in 0..n {
                let want = op(s, a[i], bb[i]);
                assert_eq!(got[i].to_f64(), want.to_f64(), "{name} lane {i}");
            }
        }

        let got = store(&b, b.neg(va));
        for i in 0..n {
            assert_eq!(got[i].to_f64(), a[i].neg().to_f64(), "neg lane {i}");
        }

        let got = store(&b, b.abs(va));
        for i in 0..n {
            assert_eq!(got[i].to_f64(), a[i].to_f64().abs(), "abs lane {i}");
        }

        let aa: Vec<T> = a.iter().map(|x| T::from_f64(x.to_f64().abs())).collect();
        let got = store(&b, b.sqrt(b.load(&aa)));
        for i in 0..n {
            assert!(rel_eq(got[i], aa[i].sqrt(), 1e-4), "sqrt lane {i}");
        }

        let got = store(&b, b.fma(va, vb, va));
        for i in 0..n {
            let want = T::from_f64(a[i].to_f64() * bb[i].to_f64() + a[i].to_f64());
            assert!(rel_eq(got[i], want, 1e-4), "fma lane {i}");
        }

        for (name, m, pred) in [
            ("le", b.le(va, vb), (|x: f64, y: f64| x <= y) as fn(f64, f64) -> bool),
            ("lt", b.lt(va, vb), |x, y| x < y),
            ("ge", b.ge(va, vb), |x, y| x >= y),
            ("gt", b.gt(va, vb), |x, y| x > y),
        ] {
            let any = b.any(m);
            let all = b.all(m);
            let want_any = (0..n).any(|i| pred(a[i].to_f64(), bb[i].to_f64()));
            let want_all = (0..n).all(|i| pred(a[i].to_f64(), bb[i].to_f64()));
            assert_eq!(any, want_any, "{name}.any");
            assert_eq!(all, want_all, "{name}.all");
        }

        let mle = b.le(va, vb);
        let got = store(&b, b.select(mle, va, vb));
        for i in 0..n {
            let want = if a[i].to_f64() <= bb[i].to_f64() { a[i] } else { bb[i] };
            assert_eq!(got[i].to_f64(), want.to_f64(), "select lane {i}");
        }
        let mnot = b.mask_not(mle);
        let got = store(&b, b.select(mnot, va, vb));
        for i in 0..n {
            let want = if a[i].to_f64() > bb[i].to_f64() { a[i] } else { bb[i] };
            assert_eq!(got[i].to_f64(), want.to_f64(), "select(!m) lane {i}");
        }

        let sum = b.reduce_sum(va).to_f64();
        let want_sum: f64 = a.iter().map(|x| x.to_f64()).sum();
        assert!((sum - want_sum).abs() <= 1e-3 * (1.0 + want_sum.abs()), "reduce_sum");
        let rmin = b.reduce_min(va).to_f64();
        let rmax = b.reduce_max(va).to_f64();
        assert_eq!(rmin, a.iter().map(|x| x.to_f64()).fold(f64::INFINITY, f64::min));
        assert_eq!(rmax, a.iter().map(|x| x.to_f64()).fold(f64::NEG_INFINITY, f64::max));
    }
}

/// A backend's register-blocked `mma` must match the scalar-oracle GEMM (tolerance: `fma` may be
/// fused). `N = 10` exercises both the lane-blocked body and the scalar tail at every lane width.
fn check_mma_one<T, S>(b: S, tol: f64)
where
    T: Scalar<Compute = T>,
    S: crate::matrix::MatrixBackend<T>,
{
    use crate::matrix::{Accumulator, Layout, MatrixA, MatrixB};
    const M: usize = 4;
    const N: usize = 10;
    const K: usize = 5;
    let mut rng = rand::rng();
    let mut draw = |n| (0..n).map(|_| T::from_f64(rng.random_range(-5.0f64..5.0))).collect::<Vec<_>>();
    let (af, bf, cf) = (draw(M * K), draw(K * N), draw(M * N));

    let row = Layout::RowMajor;
    let at = b.tile_load::<T, M, K, MatrixA>(&af, K, row);
    let bt = b.tile_load::<T, K, N, MatrixB>(&bf, N, row);
    let ct = b.tile_load::<T, M, N, Accumulator>(&cf, N, row);
    let mut got = vec![T::ZERO; M * N];
    b.tile_store(b.mma::<M, N, K>(at, bt, ct), &mut got, N, row);

    for i in 0..M {
        for j in 0..N {
            let mut want = cf[i * N + j];
            for k in 0..K {
                want = af[i * K + k].fma(bf[k * N + j], want);
            }
            assert!(rel_eq(got[i * N + j], want, tol), "mma [{i}][{j}]");
        }
    }
}

// Used by every SIMD backend that implements both f32 and f64; armv7 NEON is f32-only and checks
// f32 directly instead, so this is dead there.
#[cfg_attr(target_arch = "arm", allow(dead_code))]
fn check_all<S>(b: S)
where
    S: Backend<f32>
        + Backend<f64>
        + Copy
        + crate::matrix::MatrixBackend<f32>
        + crate::matrix::MatrixBackend<f64>,
{
    check::<f32, S>(b, |d| d() as f32);
    check::<f64, S>(b, |d| d());
    check_mma_one::<f32, S>(b, 1e-4);
    check_mma_one::<f64, S>(b, 1e-12);
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[test]
fn sse4_matches_scalar() {
    match super::sse4::Sse4::detect() {
        Some(b) => check_all(b),
        None => eprintln!("SSE4.1 unavailable; skipping"),
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[test]
fn avx1_matches_scalar() {
    match super::avx1::Avx1::detect() {
        Some(b) => check_all(b),
        None => eprintln!("AVX unavailable; skipping"),
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[test]
fn avx2_matches_scalar() {
    match super::avx2::Avx2::detect() {
        Some(b) => check_all(b),
        None => eprintln!("AVX2 unavailable; skipping"),
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[test]
fn avx512_matches_scalar() {
    match super::avx512::Avx512::detect() {
        Some(b) => check_all(b),
        None => eprintln!("AVX-512F unavailable; skipping"),
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_matches_scalar() {
    check_all(super::neon::Neon::new());
}

/// RVV at the 128-bit floor (`Rvv<16>`: f32x4 / f64x2). "V" mandates `VLEN ≥ 128`, so the 16-byte
/// token is valid on every "V" host; skips where the extension is absent.
#[cfg(target_arch = "riscv64")]
#[test]
fn rvv_matches_scalar() {
    match super::rvv::Rvv::<16>::detect() {
        Some(b) => check_all(b),
        None => eprintln!("RVV (V extension) unavailable; skipping"),
    }
}

/// armv7 A32 NEON is f32-only (no f64 vector), so it can't go through `check_all` (which needs
/// `Backend<f64>`); check the `f32` element-wise ops and the register-blocked `mma` directly. Skips
/// where NEON is absent.
#[cfg(target_arch = "arm")]
#[test]
fn neon_a32_matches_scalar() {
    match super::neon_a32::Neon::detect() {
        Some(b) => {
            check::<f32, _>(b, |d| d() as f32);
            check_mma_one::<f32, _>(b, 1e-4);
        }
        None => eprintln!("NEON unavailable; skipping"),
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[test]
fn wasm_simd128_matches_scalar() {
    check_all(super::wasm::Simd128::new());
}

#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
#[test]
fn wasm_relaxed_simd_matches_scalar() {
    check_all(super::wasm::RelaxedSimd::new());
}

/// AVX2 F16C widen path for `half::f16`: store 16-bit, compute in f32x8. For single ops the
/// widen path narrows exactly once, so it must match the scalar f16 oracle bit-for-bit;
/// `fma` (fused) and `reduce_sum` (order) use a tolerance.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[test]
fn f16_widen_matches_scalar() {
    use half::f16;

    const N: usize = 8;
    let Some(avx) = super::avx2::Avx2::detect() else {
        eprintln!("AVX2/F16C unavailable; skipping");
        return;
    };
    let s = ScalarBackend;
    let mut rng = rand::rng();
    let rand8 = |rng: &mut rand::rngs::ThreadRng| -> [f16; N] {
        core::array::from_fn(|_| f16::from_f32(rng.random_range(-10.0f32..10.0)))
    };

    let store = |v| {
        let mut o = [f16::ZERO; N];
        Backend::<f16>::store(avx, v, &mut o);
        o
    };

    for _ in 0..3000 {
        let a = rand8(&mut rng);
        let b = rand8(&mut rng);
        let va = Backend::<f16>::load(avx, &a);
        let vb = Backend::<f16>::load(avx, &b);

        let exact: [(&str, [f16; N], BinOp<f16>); 6] = [
            ("add", store(Backend::<f16>::add(avx, va, vb)), |s, x, y| s.add(x, y)),
            ("sub", store(Backend::<f16>::sub(avx, va, vb)), |s, x, y| s.sub(x, y)),
            ("mul", store(Backend::<f16>::mul(avx, va, vb)), |s, x, y| s.mul(x, y)),
            ("div", store(Backend::<f16>::div(avx, va, vb)), |s, x, y| s.div(x, y)),
            ("min", store(Backend::<f16>::min(avx, va, vb)), |s, x, y| s.min(x, y)),
            ("max", store(Backend::<f16>::max(avx, va, vb)), |s, x, y| s.max(x, y)),
        ];
        for (name, got, op) in exact {
            for i in 0..N {
                assert_eq!(got[i].to_bits(), op(s, a[i], b[i]).to_bits(), "{name} lane {i}");
            }
        }

        let aa = a.map(|x| f16::from_f32(x.to_f32().abs()));
        let got = store(Backend::<f16>::sqrt(avx, Backend::<f16>::load(avx, &aa)));
        for i in 0..N {
            assert_eq!(got[i].to_bits(), aa[i].sqrt().to_bits(), "sqrt lane {i}");
        }

        let m = Backend::<f16>::le(avx, va, vb);
        assert_eq!(Backend::<f16>::any(avx, m), (0..N).any(|i| a[i].to_f32() <= b[i].to_f32()));
        assert_eq!(Backend::<f16>::all(avx, m), (0..N).all(|i| a[i].to_f32() <= b[i].to_f32()));

        let got = store(Backend::<f16>::select(avx, m, va, vb));
        for i in 0..N {
            let want = if a[i].to_f32() <= b[i].to_f32() { a[i] } else { b[i] };
            assert_eq!(got[i].to_bits(), want.to_bits(), "select lane {i}");
        }

        let got = Backend::<f16>::reduce_sum(avx, va).to_f32();
        let want: f32 = a.iter().map(|x| x.to_f32()).sum();
        assert!((got - want).abs() <= 0.05 * (1.0 + want.abs()), "reduce_sum {got} vs {want}");
    }
}

/// Native AVX-512-FP16 backend (stable, raw-asm `v*ph`): hardware 32-wide f16 arithmetic. Runs
/// only where the CPU has `avx512fp16` (skips otherwise). Correctly-rounded ops match the scalar
/// f16 oracle exactly; div/sqrt/fma/reduce can differ by a ULP (the oracle double-rounds through
/// f32), so they use a tolerance.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[test]
fn f16_native_matches_scalar() {
    use half::f16;

    const N: usize = 32;
    let close = |a: f16, b: f16| -> bool {
        if a.to_bits() == b.to_bits() {
            return true;
        }
        let (x, y) = (a.to_f32(), b.to_f32());
        if !x.is_finite() || !y.is_finite() {
            return x == y;
        }
        (x - y).abs() <= 4.0e-3 * (1.0 + y.abs())
    };

    let Some(p) = super::avx512fp16::Avx512Fp16::detect() else {
        eprintln!("AVX-512-FP16 unavailable; skipping");
        return;
    };
    let s = ScalarBackend;
    let mut rng = rand::rng();
    assert_eq!(Backend::<f16>::lanes(p), 32);
    let rand32 = |rng: &mut rand::rngs::ThreadRng| -> [f16; N] {
        core::array::from_fn(|_| f16::from_f32(rng.random_range(-8.0f32..8.0)))
    };

    let store = |v| {
        let mut o = [f16::ZERO; N];
        Backend::<f16>::store(p, v, &mut o);
        o
    };

    for _ in 0..3000 {
        let a = rand32(&mut rng);
        let b = rand32(&mut rng);
        let va = Backend::<f16>::load(p, &a);
        let vb = Backend::<f16>::load(p, &b);

        let exact: [(&str, [f16; N], BinOp<f16>); 5] = [
            ("add", store(Backend::<f16>::add(p, va, vb)), |s, x, y| s.add(x, y)),
            ("sub", store(Backend::<f16>::sub(p, va, vb)), |s, x, y| s.sub(x, y)),
            ("mul", store(Backend::<f16>::mul(p, va, vb)), |s, x, y| s.mul(x, y)),
            ("min", store(Backend::<f16>::min(p, va, vb)), |s, x, y| s.min(x, y)),
            ("max", store(Backend::<f16>::max(p, va, vb)), |s, x, y| s.max(x, y)),
        ];
        for (name, got, op) in exact {
            for i in 0..N {
                assert_eq!(got[i].to_bits(), op(s, a[i], b[i]).to_bits(), "{name} lane {i}");
            }
        }

        let got = store(Backend::<f16>::neg(p, va));
        for i in 0..N {
            assert_eq!(got[i].to_bits(), a[i].neg().to_bits(), "neg lane {i}");
        }

        let got = store(Backend::<f16>::div(p, va, vb));
        for i in 0..N {
            assert!(close(got[i], s.div(a[i], b[i])), "div lane {i}");
        }
        let aa = a.map(|x| f16::from_f32(x.to_f32().abs()));
        let got = store(Backend::<f16>::sqrt(p, Backend::<f16>::load(p, &aa)));
        for i in 0..N {
            assert!(close(got[i], aa[i].sqrt()), "sqrt lane {i}");
        }
        let got = store(Backend::<f16>::fma(p, va, vb, va));
        for i in 0..N {
            let want = f16::from_f32(a[i].to_f32() * b[i].to_f32() + a[i].to_f32());
            assert!(close(got[i], want), "fma lane {i}");
        }

        let m = Backend::<f16>::le(p, va, vb);
        assert_eq!(Backend::<f16>::any(p, m), (0..N).any(|i| a[i].to_f32() <= b[i].to_f32()));
        assert_eq!(Backend::<f16>::all(p, m), (0..N).all(|i| a[i].to_f32() <= b[i].to_f32()));
        let got = store(Backend::<f16>::select(p, m, va, vb));
        for i in 0..N {
            let want = if a[i].to_f32() <= b[i].to_f32() { a[i] } else { b[i] };
            assert_eq!(got[i].to_bits(), want.to_bits(), "select lane {i}");
        }

        let got = Backend::<f16>::reduce_sum(p, va).to_f32();
        let want: f32 = a.iter().map(|x| x.to_f32()).sum();
        assert!((got - want).abs() <= 0.06 * (1.0 + want.abs()), "reduce_sum");
    }
}

/// NEON bf16 widen path: store 16-bit, compute in f32x4. Single ops narrow exactly once, so they
/// must match the scalar bf16 oracle bit-for-bit; `mma` (fused/order) uses a tolerance.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_bf16_matches_scalar() {
    use half::bf16;

    const N: usize = 4;
    let neon = super::neon::Neon::new();
    let s = ScalarBackend;
    let mut rng = rand::rng();
    let rand4 = |rng: &mut rand::rngs::ThreadRng| -> [bf16; N] {
        core::array::from_fn(|_| bf16::from_f32(rng.random_range(-10.0f32..10.0)))
    };
    let store = |v| {
        let mut o = [bf16::ZERO; N];
        Backend::<bf16>::store(neon, v, &mut o);
        o
    };

    for _ in 0..3000 {
        let a = rand4(&mut rng);
        let b = rand4(&mut rng);
        let va = Backend::<bf16>::load(neon, &a);
        let vb = Backend::<bf16>::load(neon, &b);

        let exact: [(&str, [bf16; N], BinOp<bf16>); 6] = [
            ("add", store(Backend::<bf16>::add(neon, va, vb)), |s, x, y| s.add(x, y)),
            ("sub", store(Backend::<bf16>::sub(neon, va, vb)), |s, x, y| s.sub(x, y)),
            ("mul", store(Backend::<bf16>::mul(neon, va, vb)), |s, x, y| s.mul(x, y)),
            ("div", store(Backend::<bf16>::div(neon, va, vb)), |s, x, y| s.div(x, y)),
            ("min", store(Backend::<bf16>::min(neon, va, vb)), |s, x, y| s.min(x, y)),
            ("max", store(Backend::<bf16>::max(neon, va, vb)), |s, x, y| s.max(x, y)),
        ];
        for (name, got, op) in exact {
            for i in 0..N {
                assert_eq!(got[i].to_bits(), op(s, a[i], b[i]).to_bits(), "{name} lane {i}");
            }
        }

        let m = Backend::<bf16>::le(neon, va, vb);
        assert_eq!(Backend::<bf16>::any(neon, m), (0..N).any(|i| a[i].to_f32() <= b[i].to_f32()));
        assert_eq!(Backend::<bf16>::all(neon, m), (0..N).all(|i| a[i].to_f32() <= b[i].to_f32()));
        let got = store(Backend::<bf16>::select(neon, m, va, vb));
        for i in 0..N {
            let want = if a[i].to_f32() <= b[i].to_f32() { a[i] } else { b[i] };
            assert_eq!(got[i].to_bits(), want.to_bits(), "select lane {i}");
        }
    }
}

/// AVX-512-BF16 element-wise backend: bf16 storage, f32x16 compute, hardware bf16↔f32 at the
/// load/store boundary. Single ops narrow exactly once (RNE), so they match the scalar bf16 oracle
/// bit-for-bit — this also checks the hardware `vcvtneps2bf16` rounds identically to `half`.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[test]
fn avx512bf16_matches_scalar() {
    use half::bf16;

    const N: usize = 16;
    let Some(p) = super::avx512bf16::Avx512Bf16::detect() else {
        eprintln!("AVX-512-BF16 unavailable; skipping");
        return;
    };
    let s = ScalarBackend;
    let mut rng = rand::rng();
    assert_eq!(Backend::<bf16>::lanes(p), 16);
    let rand16 = |rng: &mut rand::rngs::ThreadRng| -> [bf16; N] {
        core::array::from_fn(|_| bf16::from_f32(rng.random_range(-10.0f32..10.0)))
    };
    let store = |v| {
        let mut o = [bf16::ZERO; N];
        Backend::<bf16>::store(p, v, &mut o);
        o
    };

    for _ in 0..3000 {
        let a = rand16(&mut rng);
        let b = rand16(&mut rng);
        let va = Backend::<bf16>::load(p, &a);
        let vb = Backend::<bf16>::load(p, &b);

        let exact: [(&str, [bf16; N], BinOp<bf16>); 6] = [
            ("add", store(Backend::<bf16>::add(p, va, vb)), |s, x, y| s.add(x, y)),
            ("sub", store(Backend::<bf16>::sub(p, va, vb)), |s, x, y| s.sub(x, y)),
            ("mul", store(Backend::<bf16>::mul(p, va, vb)), |s, x, y| s.mul(x, y)),
            ("div", store(Backend::<bf16>::div(p, va, vb)), |s, x, y| s.div(x, y)),
            ("min", store(Backend::<bf16>::min(p, va, vb)), |s, x, y| s.min(x, y)),
            ("max", store(Backend::<bf16>::max(p, va, vb)), |s, x, y| s.max(x, y)),
        ];
        for (name, got, op) in exact {
            for i in 0..N {
                assert_eq!(got[i].to_bits(), op(s, a[i], b[i]).to_bits(), "{name} lane {i}");
            }
        }

        let m = Backend::<bf16>::le(p, va, vb);
        assert_eq!(Backend::<bf16>::any(p, m), (0..N).any(|i| a[i].to_f32() <= b[i].to_f32()));
        let got = store(Backend::<bf16>::select(p, m, va, vb));
        for i in 0..N {
            let want = if a[i].to_f32() <= b[i].to_f32() { a[i] } else { b[i] };
            assert_eq!(got[i].to_bits(), want.to_bits(), "select lane {i}");
        }
    }
}

/// AVX-512-BF16 `vdpbf16ps` matmul fast path: `Avx512`'s `mma` routes whole 16-wide column blocks
/// through the packed bf16 dot-product. `N = 16` (one full block) and odd `K = 5` (exercises the
/// pair loop plus the scalar `k`-tail) must match the f32 oracle within bf16 tolerance.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx512bf16_dpbf16_mma_matches_scalar() {
    if !is_x86_feature_detected!("avx512bf16") {
        eprintln!("AVX-512-BF16 unavailable; skipping");
        return;
    }
    match super::avx512::Avx512::detect() {
        Some(b) => check_dpbf16_mma(b),
        None => eprintln!("AVX-512 unavailable; skipping"),
    }
}

/// Generic over a single `MatrixBackend<bf16>` so the tile/`mma` calls resolve unambiguously (the
/// concrete `Avx512` also implements it for f32/f64, and the array `Tile` type is shared).
#[cfg(target_arch = "x86_64")]
fn check_dpbf16_mma<S: crate::matrix::MatrixBackend<half::bf16>>(b: S) {
    use crate::matrix::{Accumulator, Layout, MatrixA, MatrixB};
    use half::bf16;

    const M: usize = 4;
    const N: usize = 16;
    const K: usize = 5;
    let mut rng = rand::rng();
    let mut draw = |n| {
        (0..n).map(|_| bf16::from_f32(rng.random_range(-5.0f32..5.0))).collect::<Vec<_>>()
    };
    let (af, bf, cf) = (draw(M * K), draw(K * N), draw(M * N));

    let row = Layout::RowMajor;
    let at = b.tile_load::<bf16, M, K, MatrixA>(&af, K, row);
    let bt = b.tile_load::<bf16, K, N, MatrixB>(&bf, N, row);
    let cfloat = cf.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    let ct = b.tile_load::<f32, M, N, Accumulator>(&cfloat, N, row);
    let mut got = vec![0f32; M * N];
    b.tile_store(b.mma::<M, N, K>(at, bt, ct), &mut got, N, row);

    for i in 0..M {
        for j in 0..N {
            let mut want = cf[i * N + j].to_f32();
            for k in 0..K {
                want += af[i * K + k].to_f32() * bf[k * N + j].to_f32();
            }
            assert!(rel_eq(got[i * N + j], want, 5e-2), "dpbf16 mma [{i}][{j}]");
        }
    }
}

/// AMX-BF16 `tdpbf16ps` tile kernel: `D = C + A·B` for `bf16` operands into an `f32` accumulator,
/// one tile block. Odd `K = 17` exercises the zero-padded VNNI pair tail and `N = 13` a non-16
/// column count; matches the f32 oracle within bf16 tolerance. Runs only where AMX-BF16 is present
/// and tile-data permission was granted (skips otherwise).
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[test]
fn amx_bf16_mma_matches_scalar() {
    use half::bf16;
    if !crate::arch::amx::is_supported() {
        eprintln!("AMX-BF16 unavailable; skipping");
        return;
    }
    const M: usize = 12;
    const N: usize = 13;
    const K: usize = 17;
    let mut rng = rand::rng();
    let a: Vec<bf16> = (0..M * K).map(|_| bf16::from_f32(rng.random_range(-4.0f32..4.0))).collect();
    let b: Vec<bf16> = (0..K * N).map(|_| bf16::from_f32(rng.random_range(-4.0f32..4.0))).collect();
    let c0: Vec<f32> = (0..M * N).map(|_| rng.random_range(-4.0f32..4.0)).collect();

    let mut got = c0.clone();
    unsafe {
        crate::arch::amx::mma_bf16::<M, N, K>(a.as_ptr(), K, b.as_ptr(), N, got.as_mut_ptr(), N);
    }

    for i in 0..M {
        for j in 0..N {
            let mut want = c0[i * N + j];
            for k in 0..K {
                want += a[i * K + k].to_f32() * b[k * N + j].to_f32();
            }
            assert!(rel_eq(got[i * N + j], want, 5e-2), "amx mma [{i}][{j}]");
        }
    }
}

/// AMX-FP16 `tdpfp16ps` tile kernel: `D = C + A·B` for IEEE `f16` operands into an `f32`
/// accumulator, one tile block. Same odd `K = 17` / non-16 `N = 13` corners as the bf16 case;
/// matches the f32 oracle within f16's tighter tolerance. Runs only where AMX-FP16 is present and
/// tile-data permission was granted (skips otherwise — AMX-FP16 is a separate CPUID bit, so a
/// bf16-only AMX host skips this).
#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[test]
fn amx_f16_mma_matches_scalar() {
    use half::f16;
    if !crate::arch::amx::is_supported_f16() {
        eprintln!("AMX-FP16 unavailable; skipping");
        return;
    }
    const M: usize = 12;
    const N: usize = 13;
    const K: usize = 17;
    let mut rng = rand::rng();
    let a: Vec<f16> = (0..M * K).map(|_| f16::from_f32(rng.random_range(-4.0f32..4.0))).collect();
    let b: Vec<f16> = (0..K * N).map(|_| f16::from_f32(rng.random_range(-4.0f32..4.0))).collect();
    let c0: Vec<f32> = (0..M * N).map(|_| rng.random_range(-4.0f32..4.0)).collect();

    let mut got = c0.clone();
    unsafe {
        crate::arch::amx::mma_f16::<M, N, K>(a.as_ptr(), K, b.as_ptr(), N, got.as_mut_ptr(), N);
    }

    for i in 0..M {
        for j in 0..N {
            let mut want = c0[i * N + j];
            for k in 0..K {
                want += a[i * K + k].to_f32() * b[k * N + j].to_f32();
            }
            assert!(rel_eq(got[i * N + j], want, 1e-2), "amx f16 mma [{i}][{j}]");
        }
    }
}
