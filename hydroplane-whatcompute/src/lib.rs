//! What compute this machine has.
//!
//! [`cpu`] identifies the core the current thread runs on (vendor, microarchitecture, pipeline
//! limits); it is `no_std`, allocation-free and dependency-free. [`gpu`] (feature `vulkan`,
//! on by default) enumerates the Vulkan devices: architecture, shader / matrix / ray-tracing
//! unit counts, cooperative matrix shapes, memory topology (unified, resizable BAR), shader
//! features, subgroup and workgroup limits.
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub mod cpu;
#[cfg(feature = "vulkan")]
pub mod gpu;
