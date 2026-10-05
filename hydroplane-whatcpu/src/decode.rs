//! Pure tables from raw ids to core designs. No hardware access, so every arm is unit-testable
//! on any host.

use crate::{CoreType, RawId};

/// CPU designer.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum Vendor {
    /// Not identified.
    Unknown = 0,
    /// Intel.
    Intel,
    /// AMD.
    Amd,
    /// Hygon (AMD Zen licensee).
    Hygon,
    /// Zhaoxin / Centaur.
    Zhaoxin,
    /// Apple.
    Apple,
    /// Arm Ltd. (Cortex / Neoverse).
    Arm,
    /// Qualcomm (Oryon, Kryo).
    Qualcomm,
    /// Nvidia (Denver, Carmel; Grace uses Arm Neoverse).
    Nvidia,
    /// Ampere Computing.
    Ampere,
    /// Fujitsu.
    Fujitsu,
    /// Cavium / Marvell.
    Cavium,
    /// HiSilicon.
    HiSilicon,
}

impl Vendor {
    pub(crate) fn from_index(i: u8) -> Self {
        match i {
            1 => Vendor::Intel,
            2 => Vendor::Amd,
            3 => Vendor::Hygon,
            4 => Vendor::Zhaoxin,
            5 => Vendor::Apple,
            6 => Vendor::Arm,
            7 => Vendor::Qualcomm,
            8 => Vendor::Nvidia,
            9 => Vendor::Ampere,
            10 => Vendor::Fujitsu,
            11 => Vendor::Cavium,
            12 => Vendor::HiSilicon,
            _ => Vendor::Unknown,
        }
    }

    /// Vendor from the `cpuid` leaf 0 string, in register order ebx, edx, ecx.
    #[cfg_attr(not(any(target_arch = "x86", target_arch = "x86_64")), allow(dead_code))]
    pub(crate) fn from_cpuid_string(ebx: u32, edx: u32, ecx: u32) -> Self {
        let mut s = [0u8; 12];
        s[0..4].copy_from_slice(&ebx.to_le_bytes());
        s[4..8].copy_from_slice(&edx.to_le_bytes());
        s[8..12].copy_from_slice(&ecx.to_le_bytes());
        match &s {
            b"GenuineIntel" => Vendor::Intel,
            b"AuthenticAMD" => Vendor::Amd,
            b"HygonGenuine" => Vendor::Hygon,
            b"CentaurHauls" | b"  Shanghai  " => Vendor::Zhaoxin,
            _ => Vendor::Unknown,
        }
    }

    /// Vendor from the MIDR implementer byte.
    fn from_midr_implementer(imp: u8) -> Self {
        match imp {
            0x41 => Vendor::Arm,
            0x43 => Vendor::Cavium,
            0x46 => Vendor::Fujitsu,
            0x48 => Vendor::HiSilicon,
            0x4E => Vendor::Nvidia,
            0x51 => Vendor::Qualcomm,
            0x61 => Vendor::Apple,
            0xC0 => Vendor::Ampere,
            _ => Vendor::Unknown,
        }
    }
}

/// Core design. Named after the core, not the product, because the product mixes cores on
/// hybrid parts and the load/store, cache and vector behaviour follow the core.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Microarch {
    /// Not in the tables; see [`Cpu::raw`](crate::Cpu::raw).
    Unknown,

    /// Intel Sandy Bridge (2011).
    SandyBridge,
    /// Intel Ivy Bridge.
    IvyBridge,
    /// Intel Haswell.
    Haswell,
    /// Intel Broadwell.
    Broadwell,
    /// Intel Skylake client core, including Kaby/Coffee/Comet Lake.
    Skylake,
    /// Intel Skylake-SP / Cascade Lake / Cooper Lake (AVX-512, 2x64B loads).
    SkylakeServer,
    /// Intel Palm Cove (Cannon Lake).
    PalmCove,
    /// Intel Sunny Cove, Ice Lake client (one 512-bit FMA).
    SunnyCove,
    /// Intel Sunny Cove, Ice Lake-SP (two 512-bit FMAs).
    SunnyCoveServer,
    /// Intel Willow Cove (Tiger Lake).
    WillowCove,
    /// Intel Cypress Cove (Rocket Lake).
    CypressCove,
    /// Intel Golden Cove, Alder Lake P-core (AVX2 only).
    GoldenCove,
    /// Intel Golden Cove, Sapphire Rapids (AVX-512, two 512-bit FMAs).
    GoldenCoveServer,
    /// Intel Raptor Cove, Raptor Lake P-core.
    RaptorCove,
    /// Intel Raptor Cove, Emerald Rapids.
    RaptorCoveServer,
    /// Intel Redwood Cove, Meteor Lake and Arrow Lake-U P-core.
    RedwoodCove,
    /// Intel Redwood Cove, Granite Rapids.
    RedwoodCoveServer,
    /// Intel Lion Cove (Arrow Lake and Lunar Lake P-core).
    LionCove,
    /// Intel Cougar Cove (Panther Lake P-core).
    CougarCove,
    /// Intel Goldmont (Apollo Lake).
    Goldmont,
    /// Intel Goldmont Plus (Gemini Lake).
    GoldmontPlus,
    /// Intel Tremont (Jasper/Elkhart Lake).
    Tremont,
    /// Intel Gracemont (Alder/Raptor Lake E-core, Alder Lake-N).
    Gracemont,
    /// Intel Crestmont (Meteor Lake E-core, Sierra Forest).
    Crestmont,
    /// Intel Skymont (Arrow Lake and Lunar Lake E-core).
    Skymont,
    /// Intel Darkmont (Panther Lake E-core, Clearwater Forest).
    Darkmont,

    /// AMD Zen (Naples, Raven Ridge; also Hygon Dhyana).
    Zen,
    /// AMD Zen+ (Pinnacle Ridge, Picasso).
    ZenPlus,
    /// AMD Zen 2.
    Zen2,
    /// AMD Zen 3 and Zen 3+.
    Zen3,
    /// AMD Zen 4 and Zen 4c.
    Zen4,
    /// AMD Zen 5 and Zen 5c with the full 512-bit datapath (Granite Ridge, Turin, Strix Halo).
    Zen5,
    /// AMD Zen 5 with the 256-bit datapath (Strix Point, Krackan Point).
    Zen5Mobile,

    /// Apple Firestorm / Icestorm: A14, M1 family.
    AppleFirestorm,
    /// Apple Avalanche / Blizzard: A15, M2 family.
    AppleAvalanche,
    /// Apple Everest / Sawtooth: A16.
    AppleEverest,
    /// Apple "Coll": A17 Pro.
    AppleColl,
    /// Apple "Ibiza": M3.
    AppleIbiza,
    /// Apple "Lobos": M3 Pro.
    AppleLobos,
    /// Apple "Palma": M3 Max.
    ApplePalma,
    /// Apple "Donan": M4.
    AppleDonan,
    /// Apple "Brava": M4 Pro / M4 Max.
    AppleBrava,
    /// Apple "Tahiti": A18 Pro.
    AppleTahiti,
    /// Apple "Tupai": A18.
    AppleTupai,
    /// Apple "Hidra": M5.
    AppleHidra,
    /// Apple "Sotra": M5 Pro.
    AppleSotra,
    /// Apple "Thera": listed after Sotra in the SDK; product unconfirmed.
    AppleThera,
    /// Apple "Tilos": listed after Thera in the SDK; product unconfirmed.
    AppleTilos,

    /// Arm Cortex-A53.
    CortexA53,
    /// Arm Cortex-A55.
    CortexA55,
    /// Arm Cortex-A57.
    CortexA57,
    /// Arm Cortex-A72.
    CortexA72,
    /// Arm Cortex-A73.
    CortexA73,
    /// Arm Cortex-A75.
    CortexA75,
    /// Arm Cortex-A76.
    CortexA76,
    /// Arm Cortex-A77.
    CortexA77,
    /// Arm Cortex-A78 / A78C.
    CortexA78,
    /// Arm Cortex-X1 / X1C.
    CortexX1,
    /// Arm Cortex-A510.
    CortexA510,
    /// Arm Cortex-A710.
    CortexA710,
    /// Arm Cortex-X2.
    CortexX2,
    /// Arm Cortex-A715.
    CortexA715,
    /// Arm Cortex-X3.
    CortexX3,
    /// Arm Cortex-A520.
    CortexA520,
    /// Arm Cortex-A720.
    CortexA720,
    /// Arm Cortex-X4.
    CortexX4,
    /// Arm Cortex-A725.
    CortexA725,
    /// Arm Cortex-X925.
    CortexX925,
    /// Arm Neoverse N1.
    NeoverseN1,
    /// Arm Neoverse E1.
    NeoverseE1,
    /// Arm Neoverse V1.
    NeoverseV1,
    /// Arm Neoverse N2.
    NeoverseN2,
    /// Arm Neoverse V2 (Grace, Graviton 4).
    NeoverseV2,
    /// Arm Neoverse V3 (and V3AE).
    NeoverseV3,
    /// Arm Neoverse N3.
    NeoverseN3,

    /// Qualcomm Oryon (Snapdragon X, 8 Elite).
    Oryon,
    /// Nvidia Denver.
    Denver,
    /// Nvidia Carmel.
    Carmel,
    /// Ampere AmpereOne.
    AmpereOne,
    /// Fujitsu A64FX.
    A64fx,
    /// Cavium / Marvell ThunderX2.
    ThunderX2,
    /// HiSilicon TaiShan v110 (Kunpeng 920).
    TaiShanV110,
}

pub(crate) fn decode(raw: RawId) -> (Vendor, Microarch) {
    match raw {
        RawId::X86 { vendor, family, model, stepping, core } => {
            let arch = match vendor {
                Vendor::Intel => intel(family, model, stepping, core),
                Vendor::Amd => amd(family, model),
                Vendor::Hygon if family == 0x18 => Microarch::Zen,
                _ => Microarch::Unknown,
            };
            (vendor, arch)
        }
        RawId::Midr(midr) => {
            let vendor = Vendor::from_midr_implementer((midr >> 24) as u8);
            (vendor, midr_part(vendor, ((midr >> 4) & 0xFFF) as u16))
        }
        RawId::AppleFamily(f) => (Vendor::Apple, apple_family(f)),
        _ => (Vendor::Unknown, Microarch::Unknown),
    }
}

fn intel(family: u16, model: u8, _stepping: u8, core: CoreType) -> Microarch {
    use Microarch::*;
    if family != 6 {
        return Unknown;
    }
    let e = core == CoreType::Efficiency;
    match model {
        0x2A | 0x2D => SandyBridge,
        0x3A | 0x3E => IvyBridge,
        0x3C | 0x3F | 0x45 | 0x46 => Haswell,
        0x3D | 0x47 | 0x4F | 0x56 => Broadwell,
        0x4E | 0x5E | 0x8E | 0x9E | 0xA5 | 0xA6 => Skylake,
        0x55 => SkylakeServer,
        0x66 => PalmCove,
        0x7D | 0x7E => SunnyCove,
        0x6A | 0x6C => SunnyCoveServer,
        0x8C | 0x8D => WillowCove,
        0xA7 => CypressCove,
        0x8F => GoldenCoveServer,
        0x97 | 0x9A => if e { Gracemont } else { GoldenCove },
        0xCF => RaptorCoveServer,
        0xB7 | 0xBA | 0xBF => if e { Gracemont } else { RaptorCove },
        0xAD | 0xAE => RedwoodCoveServer,
        0xAA | 0xAC | 0xB5 => if e { Crestmont } else { RedwoodCove },
        0xC5 | 0xC6 | 0xBD => if e { Skymont } else { LionCove },
        0xCC => if e { Darkmont } else { CougarCove },
        0x5C | 0x5F => Goldmont,
        0x7A => GoldmontPlus,
        0x86 | 0x96 | 0x9C => Tremont,
        0xBE => Gracemont,
        0xAF => Crestmont,
        0xDD => Darkmont,
        _ => Unknown,
    }
}

fn amd(family: u16, model: u8) -> Microarch {
    use Microarch::*;
    match family {
        0x17 => match model {
            0x08 | 0x18 => ZenPlus,
            0x00..=0x2F => Zen,
            0x30..=0xFF => Zen2,
        },
        0x19 => match model {
            0x00..=0x0F | 0x20..=0x5F => Zen3,
            _ => Zen4,
        },
        0x1A => match model {
            0x20..=0x2F | 0x60..=0x6F => Zen5Mobile,
            _ => Zen5,
        },
        _ => Unknown,
    }
}

fn midr_part(vendor: Vendor, part: u16) -> Microarch {
    use Microarch::*;
    match (vendor, part) {
        (Vendor::Arm, 0xD03) => CortexA53,
        (Vendor::Arm, 0xD05) => CortexA55,
        (Vendor::Arm, 0xD07) => CortexA57,
        (Vendor::Arm, 0xD08) => CortexA72,
        (Vendor::Arm, 0xD09) => CortexA73,
        (Vendor::Arm, 0xD0A) => CortexA75,
        (Vendor::Arm, 0xD0B) => CortexA76,
        (Vendor::Arm, 0xD0C) => NeoverseN1,
        (Vendor::Arm, 0xD0D) => CortexA77,
        (Vendor::Arm, 0xD40) => NeoverseV1,
        (Vendor::Arm, 0xD41 | 0xD4B) => CortexA78,
        (Vendor::Arm, 0xD44 | 0xD4C) => CortexX1,
        (Vendor::Arm, 0xD46) => CortexA510,
        (Vendor::Arm, 0xD47) => CortexA710,
        (Vendor::Arm, 0xD48) => CortexX2,
        (Vendor::Arm, 0xD49) => NeoverseN2,
        (Vendor::Arm, 0xD4A) => NeoverseE1,
        (Vendor::Arm, 0xD4D) => CortexA715,
        (Vendor::Arm, 0xD4E) => CortexX3,
        (Vendor::Arm, 0xD4F) => NeoverseV2,
        (Vendor::Arm, 0xD80 | 0xD88) => CortexA520,
        (Vendor::Arm, 0xD81 | 0xD89) => CortexA720,
        (Vendor::Arm, 0xD82) => CortexX4,
        (Vendor::Arm, 0xD83 | 0xD84) => NeoverseV3,
        (Vendor::Arm, 0xD85) => CortexX925,
        (Vendor::Arm, 0xD87) => CortexA725,
        (Vendor::Arm, 0xD8E) => NeoverseN3,
        (Vendor::Qualcomm, 0x001) => Oryon,
        (Vendor::Apple, 0x022 | 0x023 | 0x024 | 0x025 | 0x028 | 0x029) => AppleFirestorm,
        (Vendor::Apple, 0x032 | 0x033 | 0x034 | 0x035 | 0x038 | 0x039) => AppleAvalanche,
        (Vendor::Nvidia, 0x003) => Denver,
        (Vendor::Nvidia, 0x004) => Carmel,
        (Vendor::Ampere, 0xAC3 | 0xAC4) => AmpereOne,
        (Vendor::Fujitsu, 0x001) => A64fx,
        (Vendor::Cavium, 0x0AF) => ThunderX2,
        (Vendor::HiSilicon, 0xD01) => TaiShanV110,
        _ => Unknown,
    }
}

/// `CPUFAMILY_ARM_*` from `<mach/machine.h>`.
fn apple_family(f: u32) -> Microarch {
    use Microarch::*;
    match f {
        0x1B58_8BB3 => AppleFirestorm,
        0xDA33_D83D => AppleAvalanche,
        0x8765_EDEA => AppleEverest,
        0x2876_F5B5 => AppleColl,
        0xFA33_415E => AppleIbiza,
        0x5F4D_EA93 => AppleLobos,
        0x7201_5832 => ApplePalma,
        0x6F51_29AC => AppleDonan,
        0x17D5_B93A => AppleBrava,
        0x75D4_ACB9 => AppleTahiti,
        0x2045_26D0 => AppleTupai,
        0x1D5A_87E8 => AppleHidra,
        0xF76C_5B1A => AppleSotra,
        0xAB34_5F09 => AppleThera,
        0x01D7_A72B => AppleTilos,
        _ => Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x86(vendor: Vendor, family: u16, model: u8, core: CoreType) -> Microarch {
        decode(RawId::X86 { vendor, family, model, stepping: 0, core }).1
    }

    #[test]
    fn intel_hybrid_splits_on_core_type() {
        assert_eq!(x86(Vendor::Intel, 6, 0xC6, CoreType::Performance), Microarch::LionCove);
        assert_eq!(x86(Vendor::Intel, 6, 0xC6, CoreType::Efficiency), Microarch::Skymont);
        assert_eq!(x86(Vendor::Intel, 6, 0x97, CoreType::Unspecified), Microarch::GoldenCove);
        assert_eq!(x86(Vendor::Intel, 6, 0xBE, CoreType::Unspecified), Microarch::Gracemont);
        assert_eq!(x86(Vendor::Intel, 6, 0x55, CoreType::Unspecified), Microarch::SkylakeServer);
        assert_eq!(x86(Vendor::Intel, 6, 0x8F, CoreType::Unspecified), Microarch::GoldenCoveServer);
        assert_eq!(x86(Vendor::Intel, 6, 0x7E, CoreType::Unspecified), Microarch::SunnyCove);
        assert_eq!(x86(Vendor::Intel, 6, 0x6A, CoreType::Unspecified), Microarch::SunnyCoveServer);
    }

    #[test]
    fn amd_families() {
        assert_eq!(x86(Vendor::Amd, 0x17, 0x01, CoreType::Unspecified), Microarch::Zen);
        assert_eq!(x86(Vendor::Amd, 0x17, 0x08, CoreType::Unspecified), Microarch::ZenPlus);
        assert_eq!(x86(Vendor::Amd, 0x17, 0x71, CoreType::Unspecified), Microarch::Zen2);
        assert_eq!(x86(Vendor::Amd, 0x19, 0x21, CoreType::Unspecified), Microarch::Zen3);
        assert_eq!(x86(Vendor::Amd, 0x19, 0x61, CoreType::Unspecified), Microarch::Zen4);
        assert_eq!(x86(Vendor::Amd, 0x19, 0x11, CoreType::Unspecified), Microarch::Zen4);
        assert_eq!(x86(Vendor::Amd, 0x1A, 0x44, CoreType::Unspecified), Microarch::Zen5);
        assert_eq!(x86(Vendor::Amd, 0x1A, 0x24, CoreType::Unspecified), Microarch::Zen5Mobile);
        assert_eq!(x86(Vendor::Amd, 0x1A, 0x70, CoreType::Unspecified), Microarch::Zen5);
        assert_eq!(x86(Vendor::Hygon, 0x18, 0x00, CoreType::Unspecified), Microarch::Zen);
    }

    #[test]
    fn cpuid_vendor_string() {
        assert_eq!(
            Vendor::from_cpuid_string(0x756E_6547, 0x4965_6E69, 0x6C65_746E),
            Vendor::Intel
        );
        assert_eq!(
            Vendor::from_cpuid_string(0x6874_7541, 0x6974_6E65, 0x444D_4163),
            Vendor::Amd
        );
    }

    #[test]
    fn midr_parts() {
        assert_eq!(decode(RawId::Midr(0x410F_D4F0)), (Vendor::Arm, Microarch::NeoverseV2));
        assert_eq!(decode(RawId::Midr(0x410F_D820)), (Vendor::Arm, Microarch::CortexX4));
        assert_eq!(decode(RawId::Midr(0x510F_0010)), (Vendor::Qualcomm, Microarch::Oryon));
        assert_eq!(decode(RawId::Midr(0x611F_0230)), (Vendor::Apple, Microarch::AppleFirestorm));
        assert_eq!(decode(RawId::Midr(0x0000_0000)).1, Microarch::Unknown);
    }

    #[test]
    fn apple_families() {
        assert_eq!(decode(RawId::AppleFamily(0xF76C_5B1A)), (Vendor::Apple, Microarch::AppleSotra));
        assert_eq!(decode(RawId::AppleFamily(0x6F51_29AC)).1, Microarch::AppleDonan);
        assert_eq!(decode(RawId::AppleFamily(0)).1, Microarch::Unknown);
    }
}
