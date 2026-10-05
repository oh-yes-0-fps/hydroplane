//! Per-target raw id readers. Each returns exactly one [`RawId`] with the fewest instructions
//! that pin down the core design.

use crate::RawId;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// `__cpuid` became a safe fn after this crate's MSRV; the blocks are needed on older toolchains.
#[allow(unused_unsafe)]
pub fn raw_id() -> RawId {
    use crate::{CoreType, Vendor};
    #[cfg(target_arch = "x86")]
    use core::arch::x86::__cpuid;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::__cpuid;

    // SAFETY: `cpuid` exists on every x86 rustc targets; leaves above the reported maximum
    // are only queried after checking it.
    let l0 = unsafe { __cpuid(0) };
    let vendor = Vendor::from_cpuid_string(l0.ebx, l0.edx, l0.ecx);
    let eax = unsafe { __cpuid(1) }.eax;

    let stepping = (eax & 0xF) as u8;
    let base_model = ((eax >> 4) & 0xF) as u8;
    let base_family = ((eax >> 8) & 0xF) as u16;
    let ext_model = ((eax >> 16) & 0xF) as u8;
    let ext_family = ((eax >> 20) & 0xFF) as u16;
    let family = if base_family == 0xF { base_family + ext_family } else { base_family };
    let model = if base_family == 0xF || base_family == 0x6 {
        (ext_model << 4) | base_model
    } else {
        base_model
    };

    let mut core = CoreType::Unspecified;
    if vendor == Vendor::Intel && l0.eax >= 0x1A {
        let hybrid = unsafe { __cpuid(7) }.edx & (1 << 15) != 0;
        if hybrid {
            core = match unsafe { __cpuid(0x1A) }.eax >> 24 {
                0x20 => CoreType::Efficiency,
                0x40 => CoreType::Performance,
                _ => CoreType::Unspecified,
            };
        }
    }

    RawId::X86 { vendor, family, model, stepping, core }
}

#[cfg(all(target_arch = "aarch64", any(target_os = "linux", target_os = "android")))]
pub fn raw_id() -> RawId {
    let midr: u64;
    // SAFETY: EL0 reads of MIDR_EL1 trap and are emulated by the kernel (Linux >= 4.11).
    unsafe { core::arch::asm!("mrs {}, MIDR_EL1", out(reg) midr, options(nomem, nostack, preserves_flags)) };
    RawId::Midr(midr as u32)
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
pub fn raw_id() -> RawId {
    use core::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let mut family: u32 = 0;
    let mut len = core::mem::size_of::<u32>();
    // SAFETY: libc is always linked on Apple targets; the buffer and length match.
    let rc = unsafe {
        sysctlbyname(
            c"hw.cpufamily".as_ptr(),
            (&mut family as *mut u32).cast(),
            &mut len,
            core::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && len == core::mem::size_of::<u32>() {
        RawId::AppleFamily(family)
    } else {
        RawId::Unknown
    }
}

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "aarch64", any(target_os = "linux", target_os = "android", target_vendor = "apple")),
)))]
pub fn raw_id() -> RawId {
    RawId::Unknown
}
