//! Apple GPU core count from the IORegistry: the `AGXAccelerator` service's `gpu-core-count`,
//! which neither Metal nor MoltenVK exposes.

use core::ffi::{c_char, c_void};

type CFTypeRef = *const c_void;
type IoObject = u32;

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(name: *const c_char) -> *mut c_void;
    fn IOServiceGetMatchingService(main_port: u32, matching: *mut c_void) -> IoObject;
    fn IORegistryEntryCreateCFProperty(entry: IoObject, key: CFTypeRef, allocator: CFTypeRef, options: u32) -> CFTypeRef;
    fn IOObjectRelease(object: IoObject) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(allocator: CFTypeRef, s: *const c_char, encoding: u32) -> CFTypeRef;
    fn CFNumberGetValue(number: CFTypeRef, kind: isize, out: *mut c_void) -> u8;
    fn CFGetTypeID(cf: CFTypeRef) -> usize;
    fn CFNumberGetTypeID() -> usize;
    fn CFRelease(cf: CFTypeRef);
}

const UTF8: u32 = 0x0800_0100;
const SINT64: isize = 4;

pub fn gpu_core_count() -> Option<u32> {
    // SAFETY: IOServiceGetMatchingService consumes the matching dictionary; every other object
    // created here is released once, and the property's type is checked before it is read.
    unsafe {
        let service = IOServiceGetMatchingService(0, IOServiceMatching(c"AGXAccelerator".as_ptr()));
        if service == 0 {
            return None;
        }
        let key = CFStringCreateWithCString(core::ptr::null(), c"gpu-core-count".as_ptr(), UTF8);
        let value = IORegistryEntryCreateCFProperty(service, key, core::ptr::null(), 0);
        CFRelease(key);
        IOObjectRelease(service);
        if value.is_null() {
            return None;
        }
        let mut n = 0i64;
        let ok = CFGetTypeID(value) == CFNumberGetTypeID() && CFNumberGetValue(value, SINT64, (&raw mut n).cast()) != 0;
        CFRelease(value);
        if ok { u32::try_from(n).ok().filter(|&n| n > 0) } else { None }
    }
}
