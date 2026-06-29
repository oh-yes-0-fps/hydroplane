//! Run the SVE `asm!` primitives (f32, f64, f16, bf16) and check
//! them against a scalar reference. Built static for aarch64-linux-musl and run under
//! `qemu-system-aarch64 -cpu max`, which provides base SVE (Apple silicon has SVE only via SME
//! streaming, so these can't run natively there). See `tools/qemu-sve.sh`.
#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "aarch64")]
fn main() {
    use hydroplane::arch::sve1::*;

    if !std::arch::is_aarch64_feature_detected!("sve") {
        println!("SVE_NOT_PRESENT");
        std::process::exit(2);
    }

    let mut fails = 0u32;
    let close = |g: f64, w: f64, tol: f64| (g - w).abs() <= tol + tol * w.abs();

    // f32 (C/4 = 4 lanes)
    unsafe {
        const C: usize = 16;
        const L: usize = 4;
        let a = [1.5f32, -2.0, 3.25, 4.0];
        let b = [0.5f32, 6.0, -1.0, 2.0];
        let c = [0.25f32, 0.5, 1.0, -1.0];
        let ld = |x: &[f32; L]| load_f32::<C>(x.as_ptr());
        let st = |v: &SveVec<C>| {
            let mut o = [0f32; L];
            store_f32::<C>(v, o.as_mut_ptr());
            o
        };
        let (va, vb) = (ld(&a), ld(&b));
        let cases: [(&str, [f32; L], [f32; L]); 9] = [
            ("f32 add", st(&add_f32::<C>(&va, &vb)), core::array::from_fn(|i| a[i] + b[i])),
            ("f32 sub", st(&sub_f32::<C>(&va, &vb)), core::array::from_fn(|i| a[i] - b[i])),
            ("f32 mul", st(&mul_f32::<C>(&va, &vb)), core::array::from_fn(|i| a[i] * b[i])),
            ("f32 div", st(&div_f32::<C>(&va, &vb)), core::array::from_fn(|i| a[i] / b[i])),
            ("f32 min", st(&min_f32::<C>(&va, &vb)), core::array::from_fn(|i| a[i].min(b[i]))),
            ("f32 max", st(&max_f32::<C>(&va, &vb)), core::array::from_fn(|i| a[i].max(b[i]))),
            ("f32 neg", st(&neg_f32::<C>(&va)), core::array::from_fn(|i| -a[i])),
            ("f32 fma", st(&fma_f32::<C>(&va, &vb, &ld(&c))), core::array::from_fn(|i| a[i] * b[i] + c[i])),
            ("f32 sel", st(&select_f32::<C>(&le_f32::<C>(&va, &vb), &va, &vb)),
                core::array::from_fn(|i| if a[i] <= b[i] { a[i] } else { b[i] })),
        ];
        for (name, got, want) in cases {
            for i in 0..L {
                if !close(got[i] as f64, want[i] as f64, 1e-5) {
                    println!("FAIL {name}[{i}]: {} vs {}", got[i], want[i]);
                    fails += 1;
                }
            }
        }
        let s: f32 = a.iter().sum();
        if !close(reduce_sum_f32::<C>(&va) as f64, s as f64, 1e-4) { println!("FAIL f32 reduce_sum"); fails += 1; }
        if !any_mask::<C>(&le_f32::<C>(&va, &vb)) { println!("FAIL f32 any"); fails += 1; }
    }

    // f64 (C/8 = 2 lanes — fits the 128-bit VL floor; wider VLs use a larger `C`)
    unsafe {
        const C: usize = 16;
        const L: usize = 2;
        let a = [1.5f64, -2.0];
        let b = [0.5f64, 6.0];
        let c = [0.25f64, 0.5];
        let ld = |x: &[f64; L]| load_f64::<C>(x.as_ptr());
        let st = |v: &SveVec<C>| {
            let mut o = [0f64; L];
            store_f64::<C>(v, o.as_mut_ptr());
            o
        };
        let (va, vb) = (ld(&a), ld(&b));
        let cases: [(&str, [f64; L], [f64; L]); 8] = [
            ("f64 add", st(&add_f64::<C>(&va, &vb)), core::array::from_fn(|i| a[i] + b[i])),
            ("f64 sub", st(&sub_f64::<C>(&va, &vb)), core::array::from_fn(|i| a[i] - b[i])),
            ("f64 mul", st(&mul_f64::<C>(&va, &vb)), core::array::from_fn(|i| a[i] * b[i])),
            ("f64 div", st(&div_f64::<C>(&va, &vb)), core::array::from_fn(|i| a[i] / b[i])),
            ("f64 min", st(&min_f64::<C>(&va, &vb)), core::array::from_fn(|i| a[i].min(b[i]))),
            ("f64 max", st(&max_f64::<C>(&va, &vb)), core::array::from_fn(|i| a[i].max(b[i]))),
            ("f64 fma", st(&fma_f64::<C>(&va, &vb, &ld(&c))), core::array::from_fn(|i| a[i] * b[i] + c[i])),
            ("f64 sel", st(&select_f64::<C>(&le_f64::<C>(&va, &vb), &va, &vb)),
                core::array::from_fn(|i| if a[i] <= b[i] { a[i] } else { b[i] })),
        ];
        for (name, got, want) in cases {
            for i in 0..L {
                if !close(got[i], want[i], 1e-12) {
                    println!("FAIL {name}[{i}]: {} vs {}", got[i], want[i]);
                    fails += 1;
                }
            }
        }
        let s: f64 = a.iter().sum();
        if !close(reduce_sum_f64::<C>(&va), s, 1e-12) { println!("FAIL f64 reduce_sum"); fails += 1; }
    }

    let xf = [1.5f32, -2.0, 3.25, 4.0];
    let yf = [0.5f32, 6.0, -1.0, 2.0];

    // f16 native: 16-bit compute (no widen to f32), C/2 = 4 lanes
    unsafe {
        use half::f16;
        const C: usize = 8;
        let a16: [f16; 4] = core::array::from_fn(|i| f16::from_f32(xf[i]));
        let b16: [f16; 4] = core::array::from_fn(|i| f16::from_f32(yf[i]));

        let prod = fma_f16::<C>(&load_f16::<C>(a16.as_ptr()), &load_f16::<C>(b16.as_ptr()), &splat_f16::<C>(f16::ZERO));
        let mut o16 = [f16::ZERO; 4];
        store_f16::<C>(&prod, o16.as_mut_ptr());
        for i in 0..4 {
            let want = (a16[i].to_f32() * b16[i].to_f32()) as f64;
            if !close(o16[i].to_f32() as f64, want, 2e-2) { println!("FAIL f16 fma[{i}]"); fails += 1; }
        }

        let sum = add_f16::<C>(&load_f16::<C>(a16.as_ptr()), &load_f16::<C>(b16.as_ptr()));
        let mut os = [f16::ZERO; 4];
        store_f16::<C>(&sum, os.as_mut_ptr());
        for i in 0..4 {
            let want = (a16[i].to_f32() + b16[i].to_f32()) as f64;
            if !close(os[i].to_f32() as f64, want, 2e-2) { println!("FAIL f16 add[{i}]"); fails += 1; }
        }

        let rs = reduce_sum_f16::<C>(&load_f16::<C>(a16.as_ptr())).to_f32() as f64;
        let want_rs: f64 = a16.iter().map(|x| x.to_f32() as f64).sum();
        if !close(rs, want_rs, 5e-2) { println!("FAIL f16 reduce_sum"); fails += 1; }
    }

    // bf16: f32 SVE compute (no native bf16 ALU), C/4 = 4 lanes
    unsafe {
        use half::bf16;
        const C: usize = 16;
        let ab: [bf16; 4] = core::array::from_fn(|i| bf16::from_f32(xf[i]));
        let bb: [bf16; 4] = core::array::from_fn(|i| bf16::from_f32(yf[i]));
        let sum = add_f32::<C>(&load_bf16::<C>(ab.as_ptr()), &load_bf16::<C>(bb.as_ptr()));
        let mut ob = [bf16::ZERO; 4];
        store_bf16::<C>(&sum, ob.as_mut_ptr());
        for i in 0..4 {
            let want = (ab[i].to_f32() + bb[i].to_f32()) as f64;
            if !close(ob[i].to_f32() as f64, want, 2e-2) { println!("FAIL bf16 add[{i}]"); fails += 1; }
        }
    }

    // End-to-end dispatch: `hydroplane::dispatch` must pick an `Sve<C>` token here (non-Apple aarch64),
    // not NEON. Proof: the chosen f32 backend reports `C/4` lanes = VL/4 (8 at 256-bit, 16 at
    // 512-bit), whereas NEON is always 4. Also check the dispatched result against a scalar sum.
    {
        use hydroplane::{Backend, Kernel, Simd, dispatch};

        struct SumKernel<'a> {
            xs: &'a [f32],
        }
        impl Kernel<f32> for SumKernel<'_> {
            type Output = (usize, f32);
            fn run<S: Backend<f32>>(self, simd: Simd<f32, S>) -> (usize, f32) {
                let b = simd.backend();
                let lanes = b.lanes();
                let mut acc = b.splat(0.0);
                let mut i = 0;
                while i + lanes <= self.xs.len() {
                    acc = b.add(acc, b.load(&self.xs[i..i + lanes]));
                    i += lanes;
                }
                let mut s = b.reduce_sum(acc);
                while i < self.xs.len() {
                    s += self.xs[i];
                    i += 1;
                }
                (lanes, s)
            }
        }

        let xs: Vec<f32> = (0..96).map(|i| i as f32 * 0.5 - 7.0).collect();
        let (lanes, sum) = dispatch(SumKernel { xs: &xs });
        let want: f32 = xs.iter().sum();
        if !close(sum as f64, want as f64, 1e-3) {
            println!("FAIL dispatch sum: {sum} vs {want}");
            fails += 1;
        }
        let vl = hydroplane::arch::sve2::vl_bytes();
        let c = if vl >= 64 { 64 } else if vl >= 32 { 32 } else { 16 };
        if lanes != c / 4 {
            println!("FAIL dispatch picked {lanes} f32 lanes, expected SVE {} (VL {vl}B)", c / 4);
            fails += 1;
        }
        println!("DISPATCH_LANES={lanes}");
    }

    if fails == 0 {
        println!("SVE_ALL_OK");
    } else {
        println!("SVE_FAILS={fails}");
    }
    std::process::exit(if fails == 0 { 0 } else { 1 });
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {}
