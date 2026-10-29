//! Dispatch tiers as feature bundles. Each runtime-detected backend token stands for one bundle:
//! it is only constructed when the CPU has every feature in it, and code entered through it is
//! compiled with all of them (so LLVM may also use BMI2, LZCNT, AVX-512VL… alongside the ops the
//! backend names). Bundles follow what mainstream CPUs ship together, so each tier is one real
//! class of hardware and kernels compile one copy per tier:
//!
//! - `SSE41`: x86-64-v2 (everything since Nehalem / Bulldozer).
//! - `AVX`: v2 + AVX, only Sandy/Ivy Bridge and AMD's Bulldozer family. Off unless the `avx1`
//!   feature is enabled; those CPUs run the `SSE41` tier instead.
//! - `AVX2`: x86-64-v3 (Haswell, Excavator, every Zen).
//! - `AVX512`: x86-64-v4 (Skylake-SP onward, Zen 4 onward).
//! - `AVX512BF16`: v4 + BF16 (Cooper Lake, Zen 4/5, Sapphire Rapids onward).
//! - `AVX512FP16`: v4 + BF16 + FP16 (Sapphire Rapids onward; no CPU has FP16 without BF16).
//! - `SVE`: aarch64 SVE (Nvidia Grace, Graviton 3/4); Apple and Qualcomm Oryon have none.

pub const BASE: u8 = 0;
pub const SSE41: u8 = 1;
pub const AVX: u8 = 2;
pub const AVX2: u8 = 3;
pub const AVX512: u8 = 4;
pub const AVX512FP16: u8 = 5;
pub const AVX512BF16: u8 = 6;
pub const SVE: u8 = 7;

/// The bundle table, one entry per tier: `__hp_isa!(<mode> <TIER> …)` routes through it.
///
/// - `item TIER fn …`: the item, compiled with the tier's whole bundle.
/// - `detect TIER`: whether the running CPU has the whole bundle.
/// - `names TIER`: the bundle as `&[&str]`.
/// - `kitem TIER fn …` / `kstmt TIER stmt`: `#[kernel]`'s per-tier body copy and its dispatch arm,
///   cfg'd to the tier's architecture, and dropped for tiers this build leaves out.
#[doc(hidden)]
#[macro_export]
macro_rules! __hp_isa {
    (@item [$($f:tt),*] $($i:tt)*) => {
        $(#[target_feature(enable = $f)])*
        $($i)*
    };
    (@detect [$($f:tt),*]) => {
        true $(&& ::std::arch::is_x86_feature_detected!($f))*
    };
    (@names [$($f:tt),*]) => {
        &[$($f),*]
    };
    (@kitem [$($f:tt),*] $($i:tt)*) => {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        $(#[target_feature(enable = $f)])*
        $($i)*
    };
    (@kstmt [$($f:tt),*] $($s:tt)*) => {
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        $($s)*
    };

    ($mode:ident SSE41 $($rest:tt)*) => {
        $crate::__hp_isa! { @$mode ["sse3", "ssse3", "sse4.1", "sse4.2", "popcnt"] $($rest)* }
    };
    (kitem AVX $($rest:tt)*) => { $crate::__hp_isa_avx1! { kitem $($rest)* } };
    (kstmt AVX $($rest:tt)*) => { $crate::__hp_isa_avx1! { kstmt $($rest)* } };
    ($mode:ident AVX $($rest:tt)*) => {
        $crate::__hp_isa! { @$mode ["sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx"] $($rest)* }
    };
    ($mode:ident AVX2 $($rest:tt)*) => {
        $crate::__hp_isa! { @$mode [
            "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "bmi1", "bmi2", "f16c",
            "fma", "lzcnt", "movbe"
        ] $($rest)* }
    };
    ($mode:ident AVX512 $($rest:tt)*) => {
        $crate::__hp_isa! { @$mode [
            "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "bmi1", "bmi2", "f16c",
            "fma", "lzcnt", "movbe", "avx512f", "avx512cd", "avx512bw", "avx512dq", "avx512vl"
        ] $($rest)* }
    };
    ($mode:ident AVX512BF16 $($rest:tt)*) => {
        $crate::__hp_isa! { @$mode [
            "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "bmi1", "bmi2", "f16c",
            "fma", "lzcnt", "movbe", "avx512f", "avx512cd", "avx512bw", "avx512dq", "avx512vl",
            "avx512bf16"
        ] $($rest)* }
    };
    ($mode:ident AVX512FP16 $($rest:tt)*) => {
        $crate::__hp_isa! { @$mode [
            "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "bmi1", "bmi2", "f16c",
            "fma", "lzcnt", "movbe", "avx512f", "avx512cd", "avx512bw", "avx512dq", "avx512vl",
            "avx512bf16", "avx512fp16"
        ] $($rest)* }
    };
    (item SVE $($i:tt)*) => {
        #[target_feature(enable = "sve")]
        $($i)*
    };
    (kitem SVE $($i:tt)*) => {
        #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
        #[target_feature(enable = "sve")]
        $($i)*
    };
    (kstmt SVE $($s:tt)*) => {
        #[cfg(all(target_arch = "aarch64", not(target_vendor = "apple")))]
        $($s)*
    };
}

/// `#[kernel]`'s `AVX` copies: present only with the `avx1` feature.
#[cfg(feature = "avx1")]
#[doc(hidden)]
#[macro_export]
macro_rules! __hp_isa_avx1 {
    ($mode:ident $($rest:tt)*) => {
        $crate::__hp_isa! { @$mode ["sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx"] $($rest)* }
    };
}

#[cfg(not(feature = "avx1"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __hp_isa_avx1 {
    ($($rest:tt)*) => {};
}

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "x86")))]
mod tests {
    #[test]
    fn bundle_table_matches_build_script() {
        let table: &[(&str, &[&str])] = include!("isa_bundles.in");
        let macro_names: [(&str, &[&str]); 6] = [
            ("SSE41", crate::__hp_isa!(names SSE41)),
            ("AVX", crate::__hp_isa!(names AVX)),
            ("AVX2", crate::__hp_isa!(names AVX2)),
            ("AVX512", crate::__hp_isa!(names AVX512)),
            ("AVX512BF16", crate::__hp_isa!(names AVX512BF16)),
            ("AVX512FP16", crate::__hp_isa!(names AVX512FP16)),
        ];
        assert_eq!(table, &macro_names[..]);
    }
}
