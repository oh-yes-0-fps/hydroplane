//! Identify the CPU core the current thread runs on: vendor, microarchitecture, raw id.
//! `no_std`, no dependencies, no allocation; one `cpuid` burst, one `MIDR_EL1` read, or one
//! `sysctl` per probe. [`current`] caches, [`detect`] does not. [`Microarch::pipeline`] gives
//! the core's load/store, L1D and FMA throughput limits.
#![no_std]
#![warn(missing_docs)]

mod decode;
mod pipeline;
mod probe;

pub use decode::{Microarch, Vendor};
pub use pipeline::Pipeline;

/// Which side of an Intel hybrid part (Alder Lake onward) a core is on.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum CoreType {
    /// Not a hybrid part, or the CPU does not report a core type.
    Unspecified,
    /// P-core (`cpuid` leaf 0x1A core type 0x40, "Core").
    Performance,
    /// E-core (`cpuid` leaf 0x1A core type 0x20, "Atom").
    Efficiency,
}

/// Raw hardware identity as read from the CPU, before decoding to a [`Microarch`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum RawId {
    /// `cpuid` leaf 1 display family/model/stepping (extended fields folded in) plus the
    /// hybrid core type from leaf 0x1A.
    X86 {
        /// Vendor from the leaf 0 string.
        vendor: Vendor,
        /// Display family: base family plus extended family when base is 0xF.
        family: u16,
        /// Display model: extended model folded in for family 6 and 0xF+.
        model: u8,
        /// Stepping.
        stepping: u8,
        /// P/E core type on hybrid parts.
        core: CoreType,
    },
    /// `MIDR_EL1` as read on Linux/Android aarch64.
    Midr(u32),
    /// `hw.cpufamily` from Apple's `sysctl`; values are `CPUFAMILY_ARM_*` in `<mach/machine.h>`.
    AppleFamily(u32),
    /// No probe available on this target.
    Unknown,
}

/// The identified CPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Cpu {
    /// Who made the core.
    pub vendor: Vendor,
    /// Which core design it is.
    pub arch: Microarch,
    /// What the hardware actually reported, for the cases `arch` does not cover.
    pub raw: RawId,
}

impl Cpu {
    fn from_raw(raw: RawId) -> Self {
        let (vendor, arch) = decode::decode(raw);
        Cpu { vendor, arch, raw }
    }
}

/// Probe the hardware right now. Cheap, but not free: prefer [`current`] on hot paths.
#[inline]
pub fn detect() -> Cpu {
    Cpu::from_raw(probe::raw_id())
}

/// The CPU, probed once per process and cached. On hybrid parts this is the core the first
/// call ran on.
#[inline]
pub fn current() -> Cpu {
    #[cfg(target_has_atomic = "64")]
    {
        use core::sync::atomic::{AtomicU64, Ordering};
        static CACHE: AtomicU64 = AtomicU64::new(0);
        let bits = CACHE.load(Ordering::Relaxed);
        if bits != 0 {
            return Cpu::from_raw(pack::unpack(bits));
        }
        let raw = probe::raw_id();
        CACHE.store(pack::pack(raw), Ordering::Relaxed);
        Cpu::from_raw(raw)
    }
    #[cfg(not(target_has_atomic = "64"))]
    {
        detect()
    }
}

#[cfg(target_has_atomic = "64")]
mod pack {
    use super::{CoreType, RawId, Vendor};

    const TAG_X86: u64 = 1;
    const TAG_MIDR: u64 = 2;
    const TAG_APPLE: u64 = 3;
    const TAG_UNKNOWN: u64 = 4;

    /// Layout: bits 63..56 tag, 55..48 vendor, 47..40 core type, 39..32 stepping,
    /// 31..24 model, 23..8 family; or 31..0 the 32-bit id for MIDR/Apple.
    pub fn pack(raw: RawId) -> u64 {
        match raw {
            RawId::X86 { vendor, family, model, stepping, core } => {
                (TAG_X86 << 56)
                    | ((vendor as u64) << 48)
                    | ((core as u64) << 40)
                    | ((stepping as u64) << 32)
                    | ((model as u64) << 24)
                    | ((family as u64) << 8)
            }
            RawId::Midr(v) => (TAG_MIDR << 56) | v as u64,
            RawId::AppleFamily(v) => (TAG_APPLE << 56) | v as u64,
            _ => TAG_UNKNOWN << 56,
        }
    }

    pub fn unpack(bits: u64) -> RawId {
        match bits >> 56 {
            TAG_X86 => RawId::X86 {
                vendor: Vendor::from_index((bits >> 48) as u8),
                core: match (bits >> 40) as u8 {
                    1 => CoreType::Performance,
                    2 => CoreType::Efficiency,
                    _ => CoreType::Unspecified,
                },
                stepping: (bits >> 32) as u8,
                model: (bits >> 24) as u8,
                family: (bits >> 8) as u16,
            },
            TAG_MIDR => RawId::Midr(bits as u32),
            TAG_APPLE => RawId::AppleFamily(bits as u32),
            _ => RawId::Unknown,
        }
    }

    #[cfg(test)]
    #[test]
    fn roundtrip() {
        let ids = [
            RawId::X86 {
                vendor: Vendor::Intel,
                family: 6,
                model: 0xC6,
                stepping: 2,
                core: CoreType::Efficiency,
            },
            RawId::X86 {
                vendor: Vendor::Amd,
                family: 0x1A,
                model: 0x44,
                stepping: 0,
                core: CoreType::Unspecified,
            },
            RawId::Midr(0x410F_D4F0),
            RawId::AppleFamily(0xF76C_5B1A),
            RawId::Unknown,
        ];
        for id in ids {
            assert_eq!(unpack(pack(id)), id);
            assert_ne!(pack(id), 0);
        }
    }
}
