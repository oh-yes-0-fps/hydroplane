#!/usr/bin/env bash
# Run the SVE `asm!` primitives under emulated SVE (Apple silicon has no base SVE — only SME
# streaming — so the SVE backends can't run natively; QEMU `-cpu max` provides base SVE).
#
# Requires: qemu-system-aarch64 (brew install qemu), the aarch64-unknown-linux-musl target
# (rustup target add aarch64-unknown-linux-musl), and rust-lld (bundled). No cross-gcc needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${TMPDIR:-/tmp}/spmd-qemu-sve"
mkdir -p "$WORK"; cd "$WORK"

# 1. static aarch64-linux test binary (PID1 prints SVE_ALL_OK / SVE_FAILS=N, then exits)
cargo build --release --manifest-path "$ROOT/Cargo.toml" \
  --target aarch64-unknown-linux-musl --example sve_check \
  --config 'target.aarch64-unknown-linux-musl.linker="rust-lld"'
BIN="$ROOT/target/aarch64-unknown-linux-musl/release/examples/sve_check"

# 2. initramfs with the binary as /init
rm -rf root && mkdir -p root && cp "$BIN" root/init && chmod +x root/init
(cd root && find . | cpio -o -H newc 2>/dev/null | gzip > ../initramfs.cpio.gz)

# 3. kernel (Alpine virt, EFI-stub aarch64), cached
[ -f vmlinuz-virt ] || curl -sSL -o vmlinuz-virt \
  "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/aarch64/netboot/vmlinuz-virt"

# 4. boot across vector lengths (vq = 128-bit units)
for vq in "${@:-1 2 4 8}"; do
  qemu-system-aarch64 -M virt -cpu "max,sve-max-vq=$vq" -m 512 -nographic -no-reboot \
    -kernel vmlinuz-virt -initrd initramfs.cpio.gz \
    -append "console=ttyAMA0 panic=-1 rdinit=/init" > "serial_vq$vq.log" 2>&1 &
  p=$!; for _ in $(seq 1 120); do grep -qE "SVE_ALL_OK|SVE_FAILS|panic" "serial_vq$vq.log" && break; sleep 1; done
  kill $p 2>/dev/null || true; wait $p 2>/dev/null || true
  printf "VL %5d-bit (vq=%s): %s\n" "$((vq*128))" "$vq" \
    "$(grep -oE 'SVE_ALL_OK|SVE_FAILS=[0-9]+|SVE_NOT_PRESENT' "serial_vq$vq.log" | head -1)"
done
