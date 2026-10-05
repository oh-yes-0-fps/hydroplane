//! Per-core throughput limits that matter to a SIMD kernel: memory pipes, L1D bytes per cycle,
//! vector width, FMA width. Public vendor disclosures and Chips and Cheese / numberworld
//! microbenchmarks; entries are omitted rather than guessed.

use crate::Microarch;

/// Sustained per-cycle limits of one core. Byte figures are L1D-hit bandwidth for the widest
/// native access; a wider ISA register (`vector_bits` > `datapath_bits`) is split into
/// `vector_bits / datapath_bits` native ops.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct Pipeline {
    /// Max loads per cycle, any width.
    pub loads_per_cycle: u8,
    /// Max stores per cycle, any width.
    pub stores_per_cycle: u8,
    /// Max loads plus stores in one cycle; usually the address-generation pipe count.
    pub mem_ops_per_cycle: u8,
    /// L1D load bandwidth, bytes per cycle.
    pub load_bytes_per_cycle: u16,
    /// L1D store bandwidth, bytes per cycle.
    pub store_bytes_per_cycle: u16,
    /// Widest vector register the core executes (NEON 128, AVX2 256, AVX-512 512).
    pub vector_bits: u16,
    /// Native execution and load width; ops wider than this are double pumped.
    pub datapath_bits: u16,
    /// Total FMA width issued per cycle across all vector pipes; 0 when the core has no FMA.
    pub fma_bits_per_cycle: u16,
    /// L1 data cache size in bytes (the L0 on Lion Cove).
    pub l1d_bytes: u32,
    /// Cache line size in bytes.
    pub line_bytes: u16,
}

impl Pipeline {
    /// FMA lanes per cycle for an element of `bits` bits.
    pub const fn fma_lanes_per_cycle(&self, bits: u16) -> u16 {
        self.fma_bits_per_cycle / bits
    }

    /// How many `bytes`-sized tiles the L1D can deliver per cycle, rounded down.
    pub const fn tile_loads_per_cycle(&self, bytes: u16) -> u16 {
        self.load_bytes_per_cycle / bytes
    }
}

#[allow(clippy::too_many_arguments)]
const fn p(
    loads: u8,
    stores: u8,
    mem_ops: u8,
    load_bytes: u16,
    store_bytes: u16,
    vector_bits: u16,
    datapath_bits: u16,
    fma_bits: u16,
    l1d_kib: u32,
    line: u16,
) -> Option<Pipeline> {
    Some(Pipeline {
        loads_per_cycle: loads,
        stores_per_cycle: stores,
        mem_ops_per_cycle: mem_ops,
        load_bytes_per_cycle: load_bytes,
        store_bytes_per_cycle: store_bytes,
        vector_bits,
        datapath_bits,
        fma_bits_per_cycle: fma_bits,
        l1d_bytes: l1d_kib * 1024,
        line_bytes: line,
    })
}

impl Microarch {
    /// Throughput limits for this core, or `None` when no trustworthy public figures exist.
    pub const fn pipeline(self) -> Option<Pipeline> {
        use Microarch::*;
        match self {
            SandyBridge | IvyBridge => p(2, 1, 2, 32, 16, 256, 128, 0, 32, 64),
            Haswell | Broadwell => p(2, 1, 3, 64, 32, 256, 256, 512, 32, 64),
            Skylake => p(2, 1, 3, 64, 32, 256, 256, 512, 32, 64),
            SkylakeServer => p(2, 1, 3, 128, 64, 512, 512, 1024, 32, 64),
            SunnyCove => p(2, 2, 4, 128, 64, 512, 512, 512, 48, 64),
            SunnyCoveServer => p(2, 2, 4, 128, 64, 512, 512, 1024, 48, 64),
            WillowCove | CypressCove => p(2, 2, 4, 128, 64, 512, 512, 512, 48, 64),
            GoldenCove | RaptorCove | RedwoodCove => p(3, 2, 5, 96, 64, 256, 256, 512, 48, 64),
            GoldenCoveServer | RaptorCoveServer | RedwoodCoveServer => {
                p(3, 2, 5, 128, 64, 512, 512, 1024, 48, 64)
            }
            LionCove | CougarCove => p(3, 2, 5, 96, 64, 256, 256, 512, 48, 64),
            Gracemont | Crestmont => p(2, 2, 4, 32, 32, 256, 128, 256, 32, 64),
            Skymont | Darkmont => p(3, 2, 5, 48, 32, 256, 128, 512, 32, 64),

            Zen | ZenPlus => p(2, 1, 2, 32, 16, 256, 128, 256, 32, 64),
            Zen2 => p(2, 1, 3, 64, 32, 256, 256, 512, 32, 64),
            Zen3 => p(3, 2, 3, 64, 32, 256, 256, 512, 32, 64),
            Zen4 => p(3, 2, 3, 64, 32, 512, 256, 512, 32, 64),
            Zen5 => p(4, 2, 4, 128, 64, 512, 512, 1024, 48, 64),
            Zen5Mobile => p(4, 2, 4, 64, 32, 512, 256, 512, 48, 64),

            AppleFirestorm | AppleAvalanche | AppleEverest | AppleColl | AppleIbiza | AppleLobos
            | ApplePalma | AppleDonan | AppleBrava | AppleTahiti | AppleTupai | AppleHidra
            | AppleSotra | AppleThera | AppleTilos => p(3, 2, 4, 48, 32, 128, 128, 512, 128, 128),

            CortexA76 | CortexA77 | NeoverseN1 => p(2, 2, 2, 32, 16, 128, 128, 256, 64, 64),
            CortexA78 => p(3, 2, 3, 48, 32, 128, 128, 256, 64, 64),
            CortexX1 => p(3, 2, 3, 48, 32, 128, 128, 512, 64, 64),
            NeoverseV1 => p(3, 2, 3, 48, 32, 256, 256, 512, 64, 64),
            CortexA710 | NeoverseN2 | CortexA715 | CortexA720 | CortexA725 | NeoverseN3 => {
                p(3, 2, 3, 48, 32, 128, 128, 256, 64, 64)
            }
            CortexX2 | CortexX3 | NeoverseV2 => p(3, 2, 3, 48, 32, 128, 128, 512, 64, 64),
            CortexX4 | NeoverseV3 => p(3, 2, 4, 48, 32, 128, 128, 512, 64, 64),
            CortexX925 => p(4, 2, 4, 64, 32, 128, 128, 768, 64, 64),
            Oryon => p(4, 2, 4, 64, 32, 128, 128, 512, 96, 64),
            A64fx => p(2, 1, 3, 128, 64, 512, 512, 1024, 64, 256),

            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_figures_are_consistent_with_pipe_counts() {
        let all = [
            Microarch::Zen4,
            Microarch::Zen5,
            Microarch::GoldenCove,
            Microarch::LionCove,
            Microarch::AppleSotra,
            Microarch::NeoverseV2,
            Microarch::Oryon,
        ];
        for m in all {
            let p = m.pipeline().unwrap();
            let native = p.datapath_bits / 8;
            assert!(p.load_bytes_per_cycle <= p.loads_per_cycle as u16 * native, "{m:?}");
            assert!(p.store_bytes_per_cycle <= p.stores_per_cycle as u16 * native, "{m:?}");
            assert!(p.mem_ops_per_cycle >= p.loads_per_cycle.max(p.stores_per_cycle), "{m:?}");
            assert!(p.vector_bits >= p.datapath_bits, "{m:?}");
            assert_eq!(p.fma_bits_per_cycle % p.datapath_bits, 0, "{m:?}");
        }
    }

    #[test]
    fn helpers() {
        let z5 = Microarch::Zen5.pipeline().unwrap();
        assert_eq!(z5.fma_lanes_per_cycle(32), 32);
        assert_eq!(z5.tile_loads_per_cycle(64), 2);
        let m = Microarch::AppleDonan.pipeline().unwrap();
        assert_eq!(m.fma_lanes_per_cycle(32), 16);
        assert_eq!(m.tile_loads_per_cycle(64), 0);
        assert!(Microarch::Unknown.pipeline().is_none());
        assert!(Microarch::CortexA53.pipeline().is_none());
    }
}
