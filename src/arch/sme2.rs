//! SME2 = SME(v1) + multi-vector instructions and the predicate-as-counter; the single-tile ZA
//! engine is re-exported from [`super::sme1`], and [`mma_f32_wide`] tiles a `svl/2 × svl/2`
//! output across all four `.s` ZA tiles in one streaming session.
#![allow(
    dead_code,
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::needless_range_loop
)]

use core::arch::asm;
use half::{bf16, f16};

pub use super::sme1::*;

/// 2×2 ZA-tile-grid GEMM `D = C + A·B` for one full-width element type (f32/f64). The four tiles
/// hold the quadrants split at `q = svl/E` (`E` = element bytes): `za0` = rows `[0,q)`×cols `[0,q)`,
/// `za1` = `[0,q)`×`[q,N)`, `za2` = `[q,M)`×`[0,q)`, `za3` = `[q,M)`×`[q,N)`. A-columns/B-rows/C-rows
/// move via multi-vector `LD1`/`ST1`; per-tile `FMOPA`s and `MOVA`s use single-vector low/high
/// predicates. `M,N` up to `2*q`. Below `q` it works but idles three tiles, so dispatch picks it
/// only when `M` or `N` exceeds one tile width.
macro_rules! mma_wide_2x2 {
    ($name:ident, $t:ty, $zero:expr, $sz:expr, $e:literal, $ld:literal, $st:literal,
     $cnt:literal, $open:literal) => {
        #[inline]
        pub unsafe fn $name<const M: usize, const N: usize, const K: usize>(
            a: *const $t,
            lda: usize,
            b: *const $t,
            ldb: usize,
            c: *mut $t,
            ldc: usize,
        ) {
            // Pack A column-major so each k-step's A-column is one contiguous multi-vector load.
            let mut at = [[$zero; M]; K];
            for m in 0..M {
                let row = a.add(m * lda);
                for k in 0..K {
                    at[k][m] = *row.add(k);
                }
            }
            asm!(
                $open,
                "smstart",
                concat!($cnt, " {q}"),                        // q = svl/E lanes per vector
                concat!("whilelt p0.", $e, ", xzr, {m}"),     // low rows  [0, min(M,q))
                concat!("whilelt p1.", $e, ", xzr, {n}"),     // low cols  [0, min(N,q))
                concat!("whilelt p2.", $e, ", {q}, {m}"),     // high rows [q, M)
                concat!("whilelt p3.", $e, ", {q}, {n}"),     // high cols [q, N)
                concat!("whilelt pn8.", $e, ", xzr, {m}, vlx2"), // A-column counter (M over the pair)
                concat!("whilelt pn9.", $e, ", xzr, {n}, vlx2"), // B-/C-row counter (N over the pair)
                // preload C: one multi-vector row load → low/high tiles by row half
                "mov {i}, xzr",
                "30:",
                "cmp {i}, {m}",
                "b.hs 31f",
                "madd {addr}, {i}, {ldc_b}, {c}",
                concat!($ld, " {{z0.", $e, "-z1.", $e, "}}, pn9/z, [{addr}]"),
                "cmp {i}, {q}",
                "b.hs 32f",
                "mov w12, {i:w}",
                concat!("mova za0h.", $e, "[w12, 0], p1/m, z0.", $e),
                concat!("mova za1h.", $e, "[w12, 0], p3/m, z1.", $e),
                "b 33f",
                "32:",
                "sub {t}, {i}, {q}",
                "mov w12, {t:w}",
                concat!("mova za2h.", $e, "[w12, 0], p1/m, z0.", $e),
                concat!("mova za3h.", $e, "[w12, 0], p3/m, z1.", $e),
                "33:",
                "add {i}, {i}, #1",
                "b 30b",
                "31:",
                // accumulate K rank-1 updates across the 4 tiles
                "mov {k}, xzr",
                "34:",
                "cmp {k}, {kk}",
                "b.hs 35f",
                "madd {addr}, {k}, {m_b}, {at}",
                concat!($ld, " {{z0.", $e, "-z1.", $e, "}}, pn8/z, [{addr}]"),
                "madd {addr}, {k}, {ldb_b}, {b}",
                concat!($ld, " {{z2.", $e, "-z3.", $e, "}}, pn9/z, [{addr}]"),
                concat!("fmopa za0.", $e, ", p0/m, p1/m, z0.", $e, ", z2.", $e),
                concat!("fmopa za1.", $e, ", p0/m, p3/m, z0.", $e, ", z3.", $e),
                concat!("fmopa za2.", $e, ", p2/m, p1/m, z1.", $e, ", z2.", $e),
                concat!("fmopa za3.", $e, ", p2/m, p3/m, z1.", $e, ", z3.", $e),
                "add {k}, {k}, #1",
                "b 34b",
                "35:",
                // store: gather the two column halves of each row, multi-vector store
                "mov {i}, xzr",
                "36:",
                "cmp {i}, {m}",
                "b.hs 37f",
                "cmp {i}, {q}",
                "b.hs 38f",
                "mov w12, {i:w}",
                concat!("mova z0.", $e, ", p1/m, za0h.", $e, "[w12, 0]"),
                concat!("mova z1.", $e, ", p3/m, za1h.", $e, "[w12, 0]"),
                "b 39f",
                "38:",
                "sub {t}, {i}, {q}",
                "mov w12, {t:w}",
                concat!("mova z0.", $e, ", p1/m, za2h.", $e, "[w12, 0]"),
                concat!("mova z1.", $e, ", p3/m, za3h.", $e, "[w12, 0]"),
                "39:",
                "madd {addr}, {i}, {ldc_b}, {c}",
                concat!($st, " {{z0.", $e, "-z1.", $e, "}}, pn9, [{addr}]"),
                "add {i}, {i}, #1",
                "b 36b",
                "37:",
                "smstop",
                ".arch_extension nosme2",
                m = in(reg) M,
                n = in(reg) N,
                kk = in(reg) K,
                at = in(reg) at.as_ptr(),
                b = in(reg) b,
                c = in(reg) c,
                ldc_b = in(reg) ldc * $sz,
                ldb_b = in(reg) ldb * $sz,
                m_b = in(reg) M * $sz,
                q = out(reg) _,
                i = out(reg) _,
                k = out(reg) _,
                addr = out(reg) _,
                t = out(reg) _,
                out("x12") _,
                out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _, out("z5") _,
                out("z6") _, out("z7") _, out("z8") _, out("z9") _, out("z10") _, out("z11") _,
                out("z12") _, out("z13") _, out("z14") _, out("z15") _, out("z16") _, out("z17") _,
                out("z18") _, out("z19") _, out("z20") _, out("z21") _, out("z22") _, out("z23") _,
                out("z24") _, out("z25") _, out("z26") _, out("z27") _, out("z28") _, out("z29") _,
                out("z30") _, out("z31") _,
                out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _,
                out("p6") _, out("p7") _, out("p8") _, out("p9") _, out("p10") _, out("p11") _,
                out("p12") _, out("p13") _, out("p14") _, out("p15") _,
                options(nostack),
            );
        }
    };
}

// mma_f32_wide: SME2 multi-vector f32 GEMM over a 2×2 `.s`-tile grid. `M, N ≤ svl/2`.
mma_wide_2x2!(mma_f32_wide, f32, 0.0f32, 4, "s", "ld1w", "st1w", "cntw", ".arch_extension sme2");
// mma_f64_wide: SME2 multi-vector f64 GEMM over a 2×2 `.d`-tile grid (FEAT_SME_F64F64). `M, N ≤ svl/4`.
mma_wide_2x2!(
    mma_f64_wide, f64, 0.0f64, 8, "d", "ld1d", "st1d", "cntd",
    ".arch_extension sme2\n.arch_extension sme-f64f64"
);

/// Single-session packed GEMM `D = C + Aᵀ·B`, the core of the blocked SME2 GEMM. The entire
/// `pm × pn` tile grid runs inside one `smstart`/`smstop`: streaming mode forbids NEON and most
/// non-streaming SVE, so the tile loop, C load/store, and all addressing live here in the
/// streaming-legal subset (scalar GPR + SME multi-vector). `A` and `B` arrive pre-packed into
/// contiguous `K×BLK` panels (`ap` = `pm` column panels, `bp` = `pn` row panels, `BLK = 2·VL`),
/// so each `k`-step is one contiguous multi-vector load plus a pointer bump, keeping the four
/// `FMOPA`s fed. An A row panel is reused across all `pn` column tiles (BLIS reuse). Full tiles
/// only: the caller gates `M,N` as multiples of `BLK`. `c`/`ldc_b` (row stride in bytes) is the
/// strided `M×N` output. `$shift = log2(2·VL_bytes / cnt)`: 3 for f32 (.s), 4 for f64 (.d).
/// On Apple M5 the single session is perf-neutral vs re-entering per tile (streaming-mode entry
/// is cheap there); it's kept for SME hosts where it isn't.
macro_rules! mma_grid_2x2_packed {
    ($name:ident, $e:literal, $ld:literal, $st:literal, $cnt:literal, $shift:literal, $open:literal,
     $t:ty) => {
        #[inline]
        pub unsafe fn $name(
            ap: *const $t,
            bp: *const $t,
            c: *mut $t,
            ldc_b: usize,
            pm: usize,
            pn: usize,
            k: usize,
        ) {
            asm!(
                $open,
                "smstart",
                concat!($cnt, " {q}"),
                "lsl {blk}, {q}, #1",                     // BLK = 2q
                concat!("lsl {step}, {q}, #", $shift),    // 2-VL panel / C-column stride (bytes)
                "mul {panel_b}, {kk}, {step}",            // packed panel byte size = K · 2 VL
                "mul {row_b}, {blk}, {ldc_b}",            // BLK rows of C (bytes)
                concat!("whilelt p0.", $e, ", xzr, {blk}"),
                concat!("whilelt p1.", $e, ", xzr, {blk}"),
                concat!("whilelt p2.", $e, ", {q}, {blk}"),
                concat!("whilelt p3.", $e, ", {q}, {blk}"),
                concat!("whilelt pn8.", $e, ", xzr, {blk}, vlx2"),
                concat!("whilelt pn9.", $e, ", xzr, {blk}, vlx2"),
                "mov {mi}, xzr",
                "100:",                                   // for mi in 0..pm
                "cmp {mi}, {pm}",
                "b.hs 199f",
                "mov {nj}, xzr",
                "110:",                                   // for nj in 0..pn
                "cmp {nj}, {pn}",
                "b.hs 198f",
                "madd {apw}, {mi}, {panel_b}, {ap}",      // A row panel (reused across nj)
                "madd {bpw}, {nj}, {panel_b}, {bp}",      // B column panel
                "madd {ctile}, {mi}, {row_b}, {c}",       // C tile = c + mi·row_b + nj·BLK
                "madd {ctile}, {nj}, {step}, {ctile}",
                // preload C tile into the four ZA tiles
                "mov {i}, xzr",
                "120:",
                "cmp {i}, {blk}",
                "b.hs 121f",
                "madd {row}, {i}, {ldc_b}, {ctile}",
                concat!($ld, " {{z0.", $e, "-z1.", $e, "}}, pn9/z, [{row}]"),
                "cmp {i}, {q}",
                "b.hs 122f",
                "mov w12, {i:w}",
                concat!("mova za0h.", $e, "[w12, 0], p1/m, z0.", $e),
                concat!("mova za1h.", $e, "[w12, 0], p3/m, z1.", $e),
                "b 123f",
                "122:",
                "sub {t}, {i}, {q}",
                "mov w12, {t:w}",
                concat!("mova za2h.", $e, "[w12, 0], p1/m, z0.", $e),
                concat!("mova za3h.", $e, "[w12, 0], p3/m, z1.", $e),
                "123:",
                "add {i}, {i}, #1",
                "b 120b",
                "121:",
                // K reduction over the packed panels, four independent-tile FMOPAs. Bottom-tested
                // countdown loop (`subs`/`b.ne`): the body must run at least once, which holds
                // because the caller gates on `SME_MIN_DIM` so `K ≥ 1`.
                "mov {kc}, {kk}",
                "mov {aw}, {apw}",
                "mov {bw}, {bpw}",
                "130:",
                concat!($ld, " {{z0.", $e, "-z1.", $e, "}}, pn8/z, [{aw}]"),
                concat!($ld, " {{z2.", $e, "-z3.", $e, "}}, pn9/z, [{bw}]"),
                concat!("fmopa za0.", $e, ", p0/m, p1/m, z0.", $e, ", z2.", $e),
                concat!("fmopa za1.", $e, ", p0/m, p3/m, z0.", $e, ", z3.", $e),
                concat!("fmopa za2.", $e, ", p2/m, p1/m, z1.", $e, ", z2.", $e),
                concat!("fmopa za3.", $e, ", p2/m, p3/m, z1.", $e, ", z3.", $e),
                "add {aw}, {aw}, {step}",
                "add {bw}, {bw}, {step}",
                "subs {kc}, {kc}, #1",
                "b.ne 130b",
                "131:",
                // store the C tile back
                "mov {i}, xzr",
                "140:",
                "cmp {i}, {blk}",
                "b.hs 141f",
                "cmp {i}, {q}",
                "b.hs 142f",
                "mov w12, {i:w}",
                concat!("mova z0.", $e, ", p1/m, za0h.", $e, "[w12, 0]"),
                concat!("mova z1.", $e, ", p3/m, za1h.", $e, "[w12, 0]"),
                "b 143f",
                "142:",
                "sub {t}, {i}, {q}",
                "mov w12, {t:w}",
                concat!("mova z0.", $e, ", p1/m, za2h.", $e, "[w12, 0]"),
                concat!("mova z1.", $e, ", p3/m, za3h.", $e, "[w12, 0]"),
                "143:",
                "madd {row}, {i}, {ldc_b}, {ctile}",
                concat!($st, " {{z0.", $e, "-z1.", $e, "}}, pn9, [{row}]"),
                "add {i}, {i}, #1",
                "b 140b",
                "141:",
                "add {nj}, {nj}, #1",
                "b 110b",
                "198:",
                "add {mi}, {mi}, #1",
                "b 100b",
                "199:",
                "smstop",
                ".arch_extension nosme2",
                ap = in(reg) ap,
                bp = in(reg) bp,
                c = in(reg) c,
                ldc_b = in(reg) ldc_b,
                pm = in(reg) pm,
                pn = in(reg) pn,
                kk = in(reg) k,
                q = out(reg) _,
                blk = out(reg) _,
                step = out(reg) _,
                panel_b = out(reg) _,
                row_b = out(reg) _,
                mi = out(reg) _,
                nj = out(reg) _,
                kc = out(reg) _,
                i = out(reg) _,
                t = out(reg) _,
                apw = out(reg) _,
                bpw = out(reg) _,
                ctile = out(reg) _,
                row = out(reg) _,
                aw = out(reg) _,
                bw = out(reg) _,
                out("x12") _,
                out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _, out("z5") _,
                out("z6") _, out("z7") _, out("z8") _, out("z9") _, out("z10") _, out("z11") _,
                out("z12") _, out("z13") _, out("z14") _, out("z15") _, out("z16") _, out("z17") _,
                out("z18") _, out("z19") _, out("z20") _, out("z21") _, out("z22") _, out("z23") _,
                out("z24") _, out("z25") _, out("z26") _, out("z27") _, out("z28") _, out("z29") _,
                out("z30") _, out("z31") _,
                out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _,
                out("p6") _, out("p7") _, out("p8") _, out("p9") _, out("p10") _, out("p11") _,
                out("p12") _, out("p13") _, out("p14") _, out("p15") _,
                options(nostack),
            );
        }
    };
}

mma_grid_2x2_packed!(mma_f32_grid_packed, "s", "ld1w", "st1w", "cntw", "3", ".arch_extension sme2", f32);
mma_grid_2x2_packed!(
    mma_f64_grid_packed, "d", "ld1d", "st1d", "cntd", "4",
    ".arch_extension sme2\n.arch_extension sme-f64f64", f64
);

/// Single-session blocked widening GEMM `D = C + A·B` for a 16-bit input type (`bf16`/`f16`)
/// accumulating in f32, the half-precision counterpart of [`mma_f32_grid_packed`]. `BFMOPA` / the
/// FP16-widening `FMOPA` fold a pair of `k` per instruction into the `.s` ZA tiles, so a
/// `K`-reduction is `⌈K/2⌉` matrix ops over a `svl/2 × svl/2` (`q = svl/4`) output at f32 accuracy
/// (matches the f32-accumulate reference, unlike the native f16f16 [`mma_f16_wide`]). C handling
/// is identical to the f32 grid; only the K-loop differs: `ap`/`bp` are pair-packed 16-bit panels
/// (`BLK×2` per pair-row, contiguous, one 2-vector `LD1H` + pointer bump per pair-step) and the
/// four MOPAs run on `.h` operands with all-true predicates. Full tiles only: the caller gates
/// `M,N` as multiples of `BLK = svl/2`. `c`/`ldc_b` is the strided f32 `M×N` output;
/// `pairs = ⌈K/2⌉`.
macro_rules! mma_grid_2x2_widen {
    ($name:ident, $t:ty, $mopa:literal, $open:literal) => {
        #[inline]
        pub unsafe fn $name(
            ap: *const $t,
            bp: *const $t,
            c: *mut f32,
            ldc_b: usize,
            pm: usize,
            pn: usize,
            pairs: usize,
        ) {
            asm!(
                $open,
                "smstart",
                "cntw {q}",                               // q = svl/4 (.s tile width)
                "lsl {blk}, {q}, #1",                     // BLK = 2q = svl/2
                "lsl {step}, {q}, #3",                    // pair-block (2·BLK ·h) / C-column stride = 8q bytes
                "mul {panel_b}, {pairs}, {step}",         // pair-packed panel bytes = ⌈K/2⌉ · 2·BLK ·h
                "mul {row_b}, {blk}, {ldc_b}",            // BLK rows of C (bytes)
                "ptrue p0.s",                             // all-true .s (C mova / store)
                "ptrue p1.h",                             // all-true .h (MOPA governing)
                "ptrue pn9.s",                            // all-true .s 2-vector counter (C row)
                "ptrue pn8.h",                            // all-true .h 2-vector counter (A/B pair-row)
                "mov {mi}, xzr",
                "200:",
                "cmp {mi}, {pm}",
                "b.hs 299f",
                "mov {nj}, xzr",
                "210:",
                "cmp {nj}, {pn}",
                "b.hs 298f",
                "madd {apw}, {mi}, {panel_b}, {ap}",
                "madd {bpw}, {nj}, {panel_b}, {bp}",
                "madd {ctile}, {mi}, {row_b}, {c}",
                "madd {ctile}, {nj}, {step}, {ctile}",
                // preload the f32 C tile into the four ZA.s tiles (identical to the f32 grid)
                "mov {i}, xzr",
                "220:",
                "cmp {i}, {blk}",
                "b.hs 221f",
                "madd {row}, {i}, {ldc_b}, {ctile}",
                "ld1w {{z4.s-z5.s}}, pn9/z, [{row}]",
                "cmp {i}, {q}",
                "b.hs 222f",
                "mov w12, {i:w}",
                "mova za0h.s[w12, 0], p0/m, z4.s",
                "mova za1h.s[w12, 0], p0/m, z5.s",
                "b 223f",
                "222:",
                "sub {t}, {i}, {q}",
                "mov w12, {t:w}",
                "mova za2h.s[w12, 0], p0/m, z4.s",
                "mova za3h.s[w12, 0], p0/m, z5.s",
                "223:",
                "add {i}, {i}, #1",
                "b 220b",
                "221:",
                // K reduction over ⌈K/2⌉ pair-rows: one 2-vector .h load each for A (z0/z1 =
                // low/high rows) and B (z2/z3 = low/high cols), four widening MOPAs into the
                // f32 tiles. Countdown loop like the f32 grid.
                "mov {kc}, {pairs}",
                "mov {aw}, {apw}",
                "mov {bw}, {bpw}",
                "230:",
                "ld1h {{z0.h-z1.h}}, pn8/z, [{aw}]",
                "ld1h {{z2.h-z3.h}}, pn8/z, [{bw}]",
                concat!($mopa, " za0.s, p1/m, p1/m, z0.h, z2.h"),
                concat!($mopa, " za1.s, p1/m, p1/m, z0.h, z3.h"),
                concat!($mopa, " za2.s, p1/m, p1/m, z1.h, z2.h"),
                concat!($mopa, " za3.s, p1/m, p1/m, z1.h, z3.h"),
                "add {aw}, {aw}, {step}",
                "add {bw}, {bw}, {step}",
                "subs {kc}, {kc}, #1",
                "b.ne 230b",
                "231:",
                // store the f32 C tile back (identical to the f32 grid)
                "mov {i}, xzr",
                "240:",
                "cmp {i}, {blk}",
                "b.hs 241f",
                "cmp {i}, {q}",
                "b.hs 242f",
                "mov w12, {i:w}",
                "mova z4.s, p0/m, za0h.s[w12, 0]",
                "mova z5.s, p0/m, za1h.s[w12, 0]",
                "b 243f",
                "242:",
                "sub {t}, {i}, {q}",
                "mov w12, {t:w}",
                "mova z4.s, p0/m, za2h.s[w12, 0]",
                "mova z5.s, p0/m, za3h.s[w12, 0]",
                "243:",
                "madd {row}, {i}, {ldc_b}, {ctile}",
                "st1w {{z4.s-z5.s}}, pn9, [{row}]",
                "add {i}, {i}, #1",
                "b 240b",
                "241:",
                "add {nj}, {nj}, #1",
                "b 210b",
                "298:",
                "add {mi}, {mi}, #1",
                "b 200b",
                "299:",
                "smstop",
                ".arch_extension nosme2",
                ap = in(reg) ap,
                bp = in(reg) bp,
                c = in(reg) c,
                ldc_b = in(reg) ldc_b,
                pm = in(reg) pm,
                pn = in(reg) pn,
                pairs = in(reg) pairs,
                q = out(reg) _,
                blk = out(reg) _,
                step = out(reg) _,
                panel_b = out(reg) _,
                row_b = out(reg) _,
                mi = out(reg) _,
                nj = out(reg) _,
                kc = out(reg) _,
                i = out(reg) _,
                t = out(reg) _,
                apw = out(reg) _,
                bpw = out(reg) _,
                ctile = out(reg) _,
                row = out(reg) _,
                aw = out(reg) _,
                bw = out(reg) _,
                out("x12") _,
                out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _, out("z5") _,
                out("z6") _, out("z7") _, out("z8") _, out("z9") _, out("z10") _, out("z11") _,
                out("z12") _, out("z13") _, out("z14") _, out("z15") _, out("z16") _, out("z17") _,
                out("z18") _, out("z19") _, out("z20") _, out("z21") _, out("z22") _, out("z23") _,
                out("z24") _, out("z25") _, out("z26") _, out("z27") _, out("z28") _, out("z29") _,
                out("z30") _, out("z31") _,
                out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _,
                out("p6") _, out("p7") _, out("p8") _, out("p9") _, out("p10") _, out("p11") _,
                out("p12") _, out("p13") _, out("p14") _, out("p15") _,
                options(nostack),
            );
        }
    };
}

// bf16 -> f32: BFMOPA is base SME2. f16 -> f32: the FP16-widening FMOPA needs sme-f16f16.
mma_grid_2x2_widen!(mma_bf16_grid_packed, bf16, "bfmopa", ".arch_extension sme2");
mma_grid_2x2_widen!(mma_f16_grid_packed, f16, "fmopa", ".arch_extension sme2\n.arch_extension sme-f16f16");

/// SME2 multi-vector f16 GEMM `D = C + A·B` with native f16 accumulation (FEAT_SME_F16F16), over a
/// 1×2 `.h`-tile grid. `.h` has only two ZA tiles, so the widening is in `N`: `za0.h` = cols
/// `[0,q)`, `za1.h` = cols `[q,N)`, split at `q = svl/2`. `M ≤ svl/2`, `N ≤ svl`. f16 accumulate,
/// so results don't match the f32-accumulate backends. `c: *mut f16`.
#[inline]
pub unsafe fn mma_f16_wide<const M: usize, const N: usize, const K: usize>(
    a: *const f16,
    lda: usize,
    b: *const f16,
    ldb: usize,
    c: *mut f16,
    ldc: usize,
) {
    let mut at = [[f16::ZERO; M]; K];
    for m in 0..M {
        let row = a.add(m * lda);
        for k in 0..K {
            at[k][m] = *row.add(k);
        }
    }
    asm!(
        ".arch_extension sme2\n.arch_extension sme-f16f16",
        "smstart",
        "cnth {q}",                              // q = svl/2 lanes per .h vector
        "whilelt p0.h, xzr, {m}",                // rows (M ≤ q)
        "whilelt p1.h, xzr, {n}",                // low cols  [0, min(N,q))
        "whilelt p3.h, {q}, {n}",                // high cols [q, N)
        "whilelt pn9.h, xzr, {n}, vlx2",         // N counter for the B-row / C-row pair
        // preload C: one multi-vector row → za0 (low cols) + za1 (high cols)
        "mov {i}, xzr",
        "40:",
        "cmp {i}, {m}",
        "b.hs 41f",
        "madd {addr}, {i}, {ldc_b}, {c}",
        "ld1h {{z0.h-z1.h}}, pn9/z, [{addr}]",
        "mov w12, {i:w}",
        "mova za0h.h[w12, 0], p1/m, z0.h",
        "mova za1h.h[w12, 0], p3/m, z1.h",
        "add {i}, {i}, #1",
        "b 40b",
        "41:",
        // accumulate: A column (single vector) ⊗ B row (multi-vector) into the 2 tiles
        "mov {k}, xzr",
        "42:",
        "cmp {k}, {kk}",
        "b.hs 43f",
        "madd {addr}, {k}, {m_b}, {at}",
        "ld1h {{z0.h}}, p0/z, [{addr}]",
        "madd {addr}, {k}, {ldb_b}, {b}",
        "ld1h {{z2.h-z3.h}}, pn9/z, [{addr}]",
        "fmopa za0.h, p0/m, p1/m, z0.h, z2.h",
        "fmopa za1.h, p0/m, p3/m, z0.h, z3.h",
        "add {k}, {k}, #1",
        "b 42b",
        "43:",
        // store
        "mov {i}, xzr",
        "44:",
        "cmp {i}, {m}",
        "b.hs 45f",
        "mov w12, {i:w}",
        "mova z0.h, p1/m, za0h.h[w12, 0]",
        "mova z1.h, p3/m, za1h.h[w12, 0]",
        "madd {addr}, {i}, {ldc_b}, {c}",
        "st1h {{z0.h-z1.h}}, pn9, [{addr}]",
        "add {i}, {i}, #1",
        "b 44b",
        "45:",
        "smstop",
        ".arch_extension nosme2",
        m = in(reg) M,
        n = in(reg) N,
        kk = in(reg) K,
        at = in(reg) at.as_ptr(),
        b = in(reg) b,
        c = in(reg) c,
        ldc_b = in(reg) ldc * 2,
        ldb_b = in(reg) ldb * 2,
        m_b = in(reg) M * 2,
        q = out(reg) _,
        i = out(reg) _,
        k = out(reg) _,
        addr = out(reg) _,
        out("x12") _,
        out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _, out("z5") _,
        out("z6") _, out("z7") _, out("z8") _, out("z9") _, out("z10") _, out("z11") _,
        out("z12") _, out("z13") _, out("z14") _, out("z15") _, out("z16") _, out("z17") _,
        out("z18") _, out("z19") _, out("z20") _, out("z21") _, out("z22") _, out("z23") _,
        out("z24") _, out("z25") _, out("z26") _, out("z27") _, out("z28") _, out("z29") _,
        out("z30") _, out("z31") _,
        out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _,
        out("p6") _, out("p7") _, out("p8") _, out("p9") _, out("p10") _, out("p11") _,
        out("p12") _, out("p13") _, out("p14") _, out("p15") _,
        options(nostack),
    );
}

/// SME2 multi-vector bf16 GEMM `D = C + A·B` with an `f32` accumulator, over a 2×2 `.s`-tile grid
/// (bf16 `BFMOPA` accumulates into `.s`, which has four tiles). Each instruction folds a pair of
/// `k`, so ⌈K/2⌉ ops over a `svl/2 × svl/2` (`q = svl/4`) f32 output. A/B are pair-packed and
/// loaded with multi-vector `LD1H` (`.h` counter over `2M`/`2N`); C moves as in [`mma_f32_wide`].
/// `M, N ≤ svl/2`.
#[inline]
pub unsafe fn mma_bf16_wide<const M: usize, const N: usize, const K: usize>(
    a: *const bf16,
    lda: usize,
    b: *const bf16,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    let pairs = K.div_ceil(2);
    // apack[p][i] = [A[i][2p], A[i][2p+1]] (row i's k-pair adjacent); bpack[p][j] likewise for B.
    let mut apack = [[[bf16::ZERO; 2]; M]; K];
    let mut bpack = [[[bf16::ZERO; 2]; N]; K];
    for p in 0..pairs {
        let (k0, k1) = (2 * p, 2 * p + 1);
        for i in 0..M {
            let row = a.add(i * lda);
            apack[p][i] = [*row.add(k0), if k1 < K { *row.add(k1) } else { bf16::ZERO }];
        }
        for j in 0..N {
            let r0 = b.add(k0 * ldb);
            bpack[p][j] = [*r0.add(j), if k1 < K { *b.add(k1 * ldb).add(j) } else { bf16::ZERO }];
        }
    }
    asm!(
        ".arch_extension sme2",
        "smstart",
        "cntw {q}",                              // q = svl/4 (.s tile width / row-split point)
        "lsl {q2}, {q}, #1",                     // 2q = svl/2 (.h vector width / pair-split point)
        // .s predicates for the f32 C tiles
        "whilelt p1.s, xzr, {n}",                // low cols  [0, min(N,q))
        "whilelt p3.s, {q}, {n}",                // high cols [q, N)
        "whilelt pn9.s, xzr, {n}, vlx2",         // C-row counter (N over the .s pair)
        // .h predicates for BFMOPA (over k-pairs: 2 lanes per row/col)
        "whilelt p4.h, xzr, {m2}",               // low rows  pairs
        "whilelt p5.h, {q2}, {m2}",              // high rows pairs
        "whilelt p6.h, xzr, {n2}",               // low cols  pairs
        "whilelt p7.h, {q2}, {n2}",              // high cols pairs
        "whilelt pn8.h, xzr, {m2}, vlx2",        // A pair-column counter (2M over the .h pair)
        "whilelt pn10.h, xzr, {n2}, vlx2",       // B pair-row counter (2N over the .h pair)
        // preload f32 C into the 4 .s tiles (identical to mma_f32_wide)
        "mov {i}, xzr",
        "50:",
        "cmp {i}, {m}",
        "b.hs 51f",
        "madd {addr}, {i}, {ldc_b}, {c}",
        "ld1w {{z4.s-z5.s}}, pn9/z, [{addr}]",
        "cmp {i}, {q}",
        "b.hs 52f",
        "mov w12, {i:w}",
        "mova za0h.s[w12, 0], p1/m, z4.s",
        "mova za1h.s[w12, 0], p3/m, z5.s",
        "b 53f",
        "52:",
        "sub {t}, {i}, {q}",
        "mov w12, {t:w}",
        "mova za2h.s[w12, 0], p1/m, z4.s",
        "mova za3h.s[w12, 0], p3/m, z5.s",
        "53:",
        "add {i}, {i}, #1",
        "b 50b",
        "51:",
        // accumulate ⌈K/2⌉ widening rank-2 updates across the 4 tiles
        "mov {k}, xzr",
        "54:",
        "cmp {k}, {pairs}",
        "b.hs 55f",
        "madd {addr}, {k}, {ap_b}, {apack}",     // A pair-column → z0 (low rows), z1 (high rows)
        "ld1h {{z0.h-z1.h}}, pn8/z, [{addr}]",
        "madd {addr}, {k}, {bp_b}, {bpack}",     // B pair-row → z2 (low cols), z3 (high cols)
        "ld1h {{z2.h-z3.h}}, pn10/z, [{addr}]",
        "bfmopa za0.s, p4/m, p6/m, z0.h, z2.h",
        "bfmopa za1.s, p4/m, p7/m, z0.h, z3.h",
        "bfmopa za2.s, p5/m, p6/m, z1.h, z2.h",
        "bfmopa za3.s, p5/m, p7/m, z1.h, z3.h",
        "add {k}, {k}, #1",
        "b 54b",
        "55:",
        // store f32 C (identical to mma_f32_wide)
        "mov {i}, xzr",
        "56:",
        "cmp {i}, {m}",
        "b.hs 57f",
        "cmp {i}, {q}",
        "b.hs 58f",
        "mov w12, {i:w}",
        "mova z4.s, p1/m, za0h.s[w12, 0]",
        "mova z5.s, p3/m, za1h.s[w12, 0]",
        "b 59f",
        "58:",
        "sub {t}, {i}, {q}",
        "mov w12, {t:w}",
        "mova z4.s, p1/m, za2h.s[w12, 0]",
        "mova z5.s, p3/m, za3h.s[w12, 0]",
        "59:",
        "madd {addr}, {i}, {ldc_b}, {c}",
        "st1w {{z4.s-z5.s}}, pn9, [{addr}]",
        "add {i}, {i}, #1",
        "b 56b",
        "57:",
        "smstop",
        ".arch_extension nosme2",
        m = in(reg) M,
        n = in(reg) N,
        m2 = in(reg) 2 * M,
        n2 = in(reg) 2 * N,
        pairs = in(reg) pairs,
        apack = in(reg) apack.as_ptr(),
        bpack = in(reg) bpack.as_ptr(),
        ap_b = in(reg) 4 * M,
        bp_b = in(reg) 4 * N,
        c = in(reg) c,
        ldc_b = in(reg) ldc * 4,
        q = out(reg) _,
        q2 = out(reg) _,
        i = out(reg) _,
        k = out(reg) _,
        addr = out(reg) _,
        t = out(reg) _,
        out("x12") _,
        out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _, out("z5") _,
        out("z6") _, out("z7") _, out("z8") _, out("z9") _, out("z10") _, out("z11") _,
        out("z12") _, out("z13") _, out("z14") _, out("z15") _, out("z16") _, out("z17") _,
        out("z18") _, out("z19") _, out("z20") _, out("z21") _, out("z22") _, out("z23") _,
        out("z24") _, out("z25") _, out("z26") _, out("z27") _, out("z28") _, out("z29") _,
        out("z30") _, out("z31") _,
        out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _,
        out("p6") _, out("p7") _, out("p8") _, out("p9") _, out("p10") _, out("p11") _,
        out("p12") _, out("p13") _, out("p14") _, out("p15") _,
        options(nostack),
    );
}

/// Whether the running CPU implements SME2. Linux reads `HWCAP2_SME2` from the ELF aux vector;
/// Apple reads the `hw.optional.arm.FEAT_SME2` sysctl; other OSes return `false`
/// (see [`super::sme1::is_supported`]).
#[cfg(feature = "std")]
pub fn is_supported() -> bool {
    // Cached (see `super::sme1::is_supported`): the Apple probe is a syscall, called per `mma`.
    static CACHE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
    match CACHE.load(core::sync::atomic::Ordering::Relaxed) {
        0 => {
            let v = is_supported_uncached();
            CACHE.store(1 + v as u8, core::sync::atomic::Ordering::Relaxed);
            v
        }
        c => c == 2,
    }
}

fn is_supported_uncached() -> bool {
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn getauxval(ty: core::ffi::c_ulong) -> core::ffi::c_ulong;
        }
        const AT_HWCAP2: core::ffi::c_ulong = 26;
        const HWCAP2_SME2: core::ffi::c_ulong = 1 << 37;
        unsafe { getauxval(AT_HWCAP2) & HWCAP2_SME2 != 0 }
    }
    #[cfg(target_vendor = "apple")]
    {
        super::apple_sysctl_flag(c"hw.optional.arm.FEAT_SME2")
    }
    #[cfg(not(any(target_os = "linux", target_vendor = "apple")))]
    {
        false
    }
}
