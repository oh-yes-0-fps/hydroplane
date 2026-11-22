//! Pure tables from PCI vendor / device ids (and, for Apple, the device name) to GPU designs and
//! their per-unit resources. Device id ranges follow the vendors' die allocations; per-unit
//! figures are the vendors' published ones, and are left `None` rather than guessed.

/// GPU designer, from the PCI vendor id (or the Khronos vendor id for software drivers).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum GpuVendor {
    /// Not identified; see [`super::Gpu::vendor_id`].
    Unknown,
    /// Nvidia.
    Nvidia,
    /// AMD (and ATI).
    Amd,
    /// Intel.
    Intel,
    /// Apple.
    Apple,
    /// Qualcomm (Adreno).
    Qualcomm,
    /// Arm (Mali, Immortalis).
    Arm,
    /// Imagination (PowerVR).
    Imagination,
    /// Broadcom (VideoCore).
    Broadcom,
    /// Samsung (Xclipse).
    Samsung,
    /// Mesa software rasterizers (llvmpipe / lavapipe).
    Mesa,
    /// Google (SwiftShader).
    Google,
    /// Microsoft (Dozen over D3D12, WARP).
    Microsoft,
}

impl GpuVendor {
    /// Vendor from `VkPhysicalDeviceProperties::vendorID`.
    pub const fn from_id(id: u32) -> Self {
        match id {
            0x10DE => GpuVendor::Nvidia,
            0x1002 => GpuVendor::Amd,
            0x8086 => GpuVendor::Intel,
            0x106B => GpuVendor::Apple,
            0x5143 => GpuVendor::Qualcomm,
            0x13B5 => GpuVendor::Arm,
            0x1010 => GpuVendor::Imagination,
            0x14E4 => GpuVendor::Broadcom,
            0x144D => GpuVendor::Samsung,
            0x10005 => GpuVendor::Mesa,
            0x1AE0 => GpuVendor::Google,
            0x1414 => GpuVendor::Microsoft,
            _ => GpuVendor::Unknown,
        }
    }
}

/// GPU design generation. Same generation with a different unit layout gets its own variant
/// (`Ampere` vs `AmpereGa100`, `Turing` vs `Turing16`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum GpuArch {
    /// Not in the tables.
    Unknown,

    /// Nvidia GM1xx / GM2xx.
    Maxwell,
    /// Nvidia GP10x consumer and workstation dies.
    Pascal,
    /// Nvidia GP100 (64 lanes per SM, half-rate `f64`).
    PascalGp100,
    /// Nvidia GV100: first tensor cores, no RT cores.
    Volta,
    /// Nvidia TU10x (RTX 20): tensor and RT cores.
    Turing,
    /// Nvidia TU116 / TU117 (GTX 16): Turing SMs without tensor or RT cores.
    Turing16,
    /// Nvidia GA100 (A100): 64 lanes per SM, no RT cores.
    AmpereGa100,
    /// Nvidia GA10x (RTX 30, A-series workstation).
    Ampere,
    /// Nvidia GH100 (H100, GH200).
    Hopper,
    /// Nvidia AD10x (RTX 40, L40).
    Ada,
    /// Nvidia GB100 / GB200 (B200): sm_100, no RT cores.
    BlackwellDatacenter,
    /// Nvidia GB20x (RTX 50, RTX PRO Blackwell): sm_120.
    Blackwell,

    /// AMD GCN 1 to 5 (Southern Islands through Vega, and the Vega iGPUs).
    Gcn,
    /// AMD CDNA 1 (MI100).
    Cdna1,
    /// AMD CDNA 2 (MI200).
    Cdna2,
    /// AMD CDNA 3 (MI300).
    Cdna3,
    /// AMD RDNA 1 (Navi 1x).
    Rdna1,
    /// AMD RDNA 2 (Navi 2x, Rembrandt, Van Gogh, Raphael / Granite Ridge iGPU).
    Rdna2,
    /// AMD RDNA 3 (Navi 3x, Phoenix / Hawk Point).
    Rdna3,
    /// AMD RDNA 3.5 (Strix Point, Strix Halo, Krackan).
    Rdna35,
    /// AMD RDNA 4 (Navi 4x).
    Rdna4,

    /// Intel Gen9 / Gen9.5 (Skylake through Comet Lake, Apollo / Gemini Lake).
    IntelGen9,
    /// Intel Gen11 (Ice Lake, Jasper / Elkhart Lake).
    IntelGen11,
    /// Intel Xe-LP (Tiger Lake through Raptor Lake, DG1).
    XeLp,
    /// Intel Xe-HPG (Arc A-series, Alchemist / DG2).
    XeHpg,
    /// Intel Xe-LPG (Meteor Lake, Arrow Lake).
    XeLpg,
    /// Intel Xe2 (Lunar Lake, Arc B-series Battlemage).
    Xe2,
    /// Intel Xe3 (Panther Lake).
    Xe3,

    /// Apple GPU family 7 (A14, M1).
    Apple7,
    /// Apple GPU family 8 (A15, A16, M2).
    Apple8,
    /// Apple GPU family 9 (A17 Pro, A18, M3, M4): hardware ray tracing, mesh shading.
    Apple9,
    /// Apple GPU family 10 (A19, M5): a neural accelerator in every GPU core.
    Apple10,
}

/// Resources of one shader unit (Nvidia SM, AMD CU, Intel EU or Xe-core, Apple GPU core).
/// `None` means not publicly documented, `Some(0)` that the unit has none.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct UnitLayout {
    /// What the vendor calls the unit.
    pub name: &'static str,
    /// FP32 lanes (CUDA cores, stream processors, ALUs). AMD RDNA 3+ dual-issue is not counted.
    pub fp32_lanes: u16,
    /// Matrix engines: tensor cores (Nvidia), matrix cores / AI accelerators (AMD), XMX engines
    /// (Intel), neural accelerators (Apple).
    pub matrix_units: Option<u8>,
    /// Ray-tracing units: RT cores (Nvidia), ray accelerators (AMD), RTUs (Intel).
    pub rt_units: Option<u8>,
}

const fn u(name: &'static str, fp32_lanes: u16, matrix: u8, rt: u8) -> Option<UnitLayout> {
    Some(UnitLayout { name, fp32_lanes, matrix_units: Some(matrix), rt_units: Some(rt) })
}

impl GpuArch {
    /// Per-unit resources, or `None` for designs without trustworthy public figures.
    pub const fn unit_layout(self) -> Option<UnitLayout> {
        use GpuArch::*;
        match self {
            Maxwell | Pascal => u("SM", 128, 0, 0),
            PascalGp100 => u("SM", 64, 0, 0),
            Volta => u("SM", 64, 8, 0),
            Turing => u("SM", 64, 8, 1),
            Turing16 => u("SM", 64, 0, 0),
            AmpereGa100 => u("SM", 64, 4, 0),
            Ampere | Ada | Blackwell => u("SM", 128, 4, 1),
            Hopper | BlackwellDatacenter => u("SM", 128, 4, 0),

            Gcn | Rdna1 => u("CU", 64, 0, 0),
            Cdna1 | Cdna2 | Cdna3 => u("CU", 64, 4, 0),
            Rdna2 => u("CU", 64, 0, 1),
            Rdna3 | Rdna35 | Rdna4 => u("CU", 64, 2, 1),

            IntelGen9 | IntelGen11 | XeLp => u("EU", 8, 0, 0),
            XeHpg => u("Xe-core", 128, 16, 1),
            XeLpg => u("Xe-core", 128, 0, 1),
            Xe2 | Xe3 => u("Xe-core", 128, 8, 1),

            Apple7 | Apple8 => u("core", 128, 0, 0),
            Apple9 => Some(UnitLayout { name: "core", fp32_lanes: 128, matrix_units: Some(0), rt_units: None }),
            Apple10 => Some(UnitLayout { name: "core", fp32_lanes: 128, matrix_units: Some(1), rt_units: None }),

            Unknown => None,
        }
    }

    /// Whether the design has fixed-function ray-tracing hardware, when known.
    pub const fn ray_tracing_hardware(self) -> Option<bool> {
        match self {
            GpuArch::Apple9 | GpuArch::Apple10 => Some(true),
            _ => match self.unit_layout() {
                Some(UnitLayout { rt_units: Some(n), .. }) => Some(n > 0),
                _ => None,
            },
        }
    }

    /// Whether the design has dedicated matrix-multiply hardware, when known.
    pub const fn matrix_hardware(self) -> Option<bool> {
        match self.unit_layout() {
            Some(UnitLayout { matrix_units: Some(n), .. }) => Some(n > 0),
            _ => None,
        }
    }
}

pub(crate) fn decode(vendor: GpuVendor, device_id: u32, name: &str) -> GpuArch {
    match vendor {
        GpuVendor::Nvidia => nvidia(device_id),
        GpuVendor::Amd => amd(device_id),
        GpuVendor::Intel => intel(device_id),
        GpuVendor::Apple => apple(name),
        _ => GpuArch::Unknown,
    }
}

fn nvidia(id: u32) -> GpuArch {
    use GpuArch::*;
    match id {
        0x1340..=0x143F | 0x17C0..=0x17FF => Maxwell,
        0x15F0..=0x15FF => PascalGp100,
        0x1B00..=0x1D7F => Pascal,
        0x1D80..=0x1DFF => Volta,
        0x1E00..=0x1F7F => Turing,
        0x1F80..=0x1FFF | 0x2180..=0x21FF => Turing16,
        0x20B0..=0x20FF => AmpereGa100,
        0x2200..=0x22FF | 0x2400..=0x25FF => Ampere,
        0x2300..=0x23FF => Hopper,
        0x2600..=0x28FF => Ada,
        0x2900..=0x29FF => BlackwellDatacenter,
        0x2B00..=0x2FFF => Blackwell,
        _ => Unknown,
    }
}

fn amd(id: u32) -> GpuArch {
    use GpuArch::*;
    match id {
        0x6600..=0x69FF | 0x7300..=0x730F | 0x9800..=0x98FF => Gcn,
        0x15D8 | 0x15DD | 0x15E7 | 0x1636 | 0x1638 | 0x164C => Gcn,
        0x7388 | 0x738C | 0x738E => Cdna1,
        0x7408 | 0x740C | 0x740F | 0x7410 => Cdna2,
        0x74A0..=0x74BF => Cdna3,
        0x7310..=0x731F | 0x7340..=0x734F | 0x7360..=0x736F => Rdna1,
        0x73A0..=0x73FF | 0x7420..=0x743F => Rdna2,
        0x163F | 0x1681 | 0x164E | 0x1506 | 0x13C0 => Rdna2,
        0x7440..=0x749F | 0x15BF | 0x15C8 | 0x1900 | 0x1901 => Rdna3,
        0x150E | 0x1586 | 0x1114 => Rdna35,
        0x7550..=0x755F | 0x7590..=0x759F => Rdna4,
        _ => Unknown,
    }
}

fn intel(id: u32) -> GpuArch {
    use GpuArch::*;
    match id {
        0x1900..=0x19FF | 0x5900..=0x59FF | 0x3E00..=0x3EFF | 0x9B00..=0x9BFF => IntelGen9,
        0x87C0..=0x87CF | 0x5A80..=0x5AFF | 0x3180..=0x31FF => IntelGen9,
        0x8A50..=0x8A7F | 0x4E50..=0x4E7F => IntelGen11,
        0x9A40..=0x9AFF | 0x4C80..=0x4CFF | 0x4600..=0x46FF | 0x4900..=0x490F | 0xA700..=0xA7FF => XeLp,
        0x5690..=0x56CF => XeHpg,
        0x7D40..=0x7DFF => XeLpg,
        0x6420..=0x64FF | 0xE200..=0xE2FF => Xe2,
        0xB080..=0xB0FF => Xe3,
        _ => Unknown,
    }
}

/// From the marketing name ("Apple M5 Pro", "Apple A17 Pro GPU"): MoltenVK's device ids are
/// synthesized, not PCI ids.
fn apple(name: &str) -> GpuArch {
    use GpuArch::*;
    let Some(rest) = name.strip_prefix("Apple ") else { return Unknown };
    let mut chars = rest.chars();
    let line = chars.next();
    let digits: String = chars.take_while(char::is_ascii_digit).collect();
    let Ok(n) = digits.parse::<u32>() else { return Unknown };
    match (line, n) {
        (Some('M'), 1) | (Some('A'), 14) => Apple7,
        (Some('M'), 2) | (Some('A'), 15 | 16) => Apple8,
        (Some('M'), 3 | 4) | (Some('A'), 17 | 18) => Apple9,
        (Some('M'), 5) | (Some('A'), 19) => Apple10,
        _ => Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_ids() {
        let cases = [
            (0x1E04, GpuArch::Turing),
            (0x1F82, GpuArch::Turing16),
            (0x2182, GpuArch::Turing16),
            (0x20B0, GpuArch::AmpereGa100),
            (0x2204, GpuArch::Ampere),
            (0x2484, GpuArch::Ampere),
            (0x2330, GpuArch::Hopper),
            (0x2684, GpuArch::Ada),
            (0x2901, GpuArch::BlackwellDatacenter),
            (0x2B85, GpuArch::Blackwell),
            (0x2C02, GpuArch::Blackwell),
        ];
        for (id, arch) in cases {
            assert_eq!(decode(GpuVendor::Nvidia, id, ""), arch, "{id:#x}");
        }
    }

    #[test]
    fn amd_ids() {
        let cases = [
            (0x73BF, GpuArch::Rdna2),
            (0x744C, GpuArch::Rdna3),
            (0x747E, GpuArch::Rdna3),
            (0x74A1, GpuArch::Cdna3),
            (0x1586, GpuArch::Rdna35),
            (0x7550, GpuArch::Rdna4),
            (0x687F, GpuArch::Gcn),
        ];
        for (id, arch) in cases {
            assert_eq!(decode(GpuVendor::Amd, id, ""), arch, "{id:#x}");
        }
    }

    #[test]
    fn intel_ids() {
        assert_eq!(decode(GpuVendor::Intel, 0x56A0, ""), GpuArch::XeHpg);
        assert_eq!(decode(GpuVendor::Intel, 0xE20B, ""), GpuArch::Xe2);
        assert_eq!(decode(GpuVendor::Intel, 0x7D55, ""), GpuArch::XeLpg);
        assert_eq!(decode(GpuVendor::Intel, 0x9A49, ""), GpuArch::XeLp);
    }

    #[test]
    fn apple_names() {
        assert_eq!(decode(GpuVendor::Apple, 0, "Apple M1 Max"), GpuArch::Apple7);
        assert_eq!(decode(GpuVendor::Apple, 0, "Apple M2"), GpuArch::Apple8);
        assert_eq!(decode(GpuVendor::Apple, 0, "Apple M4 Pro"), GpuArch::Apple9);
        assert_eq!(decode(GpuVendor::Apple, 0, "Apple M5 Pro"), GpuArch::Apple10);
        assert_eq!(decode(GpuVendor::Apple, 0, "Apple A17 Pro GPU"), GpuArch::Apple9);
        assert_eq!(decode(GpuVendor::Apple, 0, "Apple Paravirtual device"), GpuArch::Unknown);
    }

    #[test]
    fn hardware_flags() {
        assert_eq!(GpuArch::Turing16.matrix_hardware(), Some(false));
        assert_eq!(GpuArch::Ada.ray_tracing_hardware(), Some(true));
        assert_eq!(GpuArch::Hopper.ray_tracing_hardware(), Some(false));
        assert_eq!(GpuArch::Apple9.ray_tracing_hardware(), Some(true));
        assert_eq!(GpuArch::Apple10.matrix_hardware(), Some(true));
        assert_eq!(GpuArch::Unknown.matrix_hardware(), None);
    }
}
