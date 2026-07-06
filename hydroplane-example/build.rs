//! Build-time MIR analysis of this crate's `#[kernel]`s; the pipeline lives in `hydroplane-auto`.
fn main() {
    hydroplane_auto::build_script();
}
