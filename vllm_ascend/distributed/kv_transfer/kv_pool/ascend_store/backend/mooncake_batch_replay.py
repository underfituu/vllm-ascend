import argparse
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from mooncake.store import MooncakeDistributedStore, ReplicateConfig  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _buffer_bytes(size: int):
    if torch is not None:
        return torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
    return ctypes.create_string_buffer(size)


def _buffer_ptr(buffer: Any) -> int:
    if torch is not None and hasattr(buffer, "data_ptr"):
        return int(buffer.data_ptr())
    return ctypes.addressof(buffer)


def _write_buffer(buffer: Any, payload: bytes) -> None:
    ptr = _buffer_ptr(buffer)
    ctypes.memmove(ptr, payload, len(payload))


def _read_buffer_bytes(buffer: Any, size: int) -> bytes:
    ptr = _buffer_ptr(buffer)
    return ctypes.string_at(ptr, size)


def _parse_size(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return int(value)
    cleaned = value.strip().lower()
    if not cleaned:
        return 0
    multipliers = {"gb": 1024**3, "mb": 1024**2, "kb": 1024, "b": 1}
    for unit, multiplier in multipliers.items():
        if cleaned.endswith(unit):
            return int(float(cleaned[: -len(unit)].strip()) * multiplier)
    return int(float(cleaned))


def _setup_store(config_path: str, local_hostname: str | None):
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    store = MooncakeDistributedStore()
    hostname = local_hostname or os.getenv("MOONCAKE_REPLAY_LOCAL_HOSTNAME", "127.0.0.1")
    ret = store.setup(
        local_hostname=hostname,
        metadata_server=config.get("metadata_server"),
        global_segment_size=_parse_size(config.get("global_segment_size", 0)),
        local_buffer_size=_parse_size(config.get("local_buffer_size", 0)),
        protocol=config.get("protocol", "ascend"),
        rdma_devices=config.get("device_name", ""),
        master_server_addr=config.get("master_server_address"),
    )
    if ret != 0:
        raise RuntimeError(f"MooncakeDistributedStore.setup failed: {ret}")
    return store


def _materialize_buffers(manifest: dict[str, Any], manifest_dir: Path):
    ptrs: list[int] = []
    lengths: list[int] = []
    addrs: list[list[int]] = []
    sizes: list[list[int]] = []
    payloads: list[bytes] = []
    holders: list[Any] = []

    for item in manifest["items"]:
        payload_path = manifest_dir / item["payload_file"]
        payload = payload_path.read_bytes()
        payloads.append(payload)
        expected_digest = item.get("payload_sha256")
        actual_digest = hashlib.sha256(payload).hexdigest()
        if expected_digest and expected_digest != actual_digest:
            raise ValueError(
                f"Payload digest mismatch for {payload_path}: {actual_digest} != {expected_digest}"
            )

        group_sizes = [int(size) for size in item["sizes"]]
        group_addrs: list[int] = []
        offset = 0
        for size in group_sizes:
            holder = _buffer_bytes(size)
            chunk = payload[offset : offset + size]
            _write_buffer(holder, chunk)
            ptr = _buffer_ptr(holder)
            holders.append(holder)
            ptrs.append(ptr)
            lengths.append(size)
            group_addrs.append(ptr)
            offset += size
        addrs.append(group_addrs)
        sizes.append(group_sizes)

    return {
        "ptrs": ptrs,
        "lengths": lengths,
        "addrs": addrs,
        "sizes": sizes,
        "payloads": payloads,
        "holders": holders,
    }


def _register_buffers(store: Any, ptrs: list[int], lengths: list[int]) -> None:
    for ptr, length in zip(ptrs, lengths, strict=True):
        ret = store.register_buffer(ptr, length)
        if ret != 0:
            raise RuntimeError(f"register_buffer failed for ptr={ptr} length={length}: {ret}")


def _run_put(store: Any, manifest: dict[str, Any], buffers: dict[str, Any]) -> list[int]:
    config = ReplicateConfig()
    start_time = time.perf_counter()
    result = store.batch_put_from_multi_buffers(
        [item["key"] for item in manifest["items"]],
        buffers["addrs"],
        buffers["sizes"],
        config,
    )
    duration_ms = (time.perf_counter() - start_time) * 1000
    print(f"PUT duration_ms={duration_ms:.3f} result={list(result)}")
    return list(result)


def _run_get(store: Any, manifest: dict[str, Any], buffers: dict[str, Any]) -> list[int]:
    for holder in buffers["holders"]:
        size = len(holder) if not hasattr(holder, "numel") else int(holder.numel())
        ctypes.memset(_buffer_ptr(holder), 0, size)
    start_time = time.perf_counter()
    result = store.batch_get_into_multi_buffers(
        [item["key"] for item in manifest["items"]],
        buffers["addrs"],
        buffers["sizes"],
    )
    duration_ms = (time.perf_counter() - start_time) * 1000
    print(f"GET duration_ms={duration_ms:.3f} result={list(result)}")
    return list(result)


def _verify_get_payloads(manifest: dict[str, Any], buffers: dict[str, Any]) -> None:
    cursor = 0
    for item, original_payload in zip(manifest["items"], buffers["payloads"], strict=True):
        reconstructed = bytearray()
        for size in item["sizes"]:
            holder = buffers["holders"][cursor]
            reconstructed.extend(_read_buffer_bytes(holder, int(size)))
            cursor += 1
        digest = hashlib.sha256(bytes(reconstructed)).hexdigest()
        expected = hashlib.sha256(original_payload).hexdigest()
        if digest != expected:
            raise AssertionError(f"Payload mismatch for key={item['key']}: {digest} != {expected}")
    print("Payload verification passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay captured Mooncake put/get batches without vLLM")
    parser.add_argument("manifest", help="Path to captured manifest json")
    parser.add_argument("--config", required=True, help="Path to Mooncake config json")
    parser.add_argument(
        "--mode",
        choices=["put", "get", "both"],
        default="both",
        help="Replay only put, only get, or put then get",
    )
    parser.add_argument(
        "--local-hostname",
        default=None,
        help="Override local_hostname passed to MooncakeDistributedStore.setup",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    store = _setup_store(args.config, args.local_hostname)
    buffers = _materialize_buffers(manifest, manifest_path.parent)
    _register_buffers(store, buffers["ptrs"], buffers["lengths"])

    if args.mode in ("put", "both"):
        _run_put(store, manifest, buffers)
    if args.mode in ("get", "both"):
        _run_get(store, manifest, buffers)
        _verify_get_payloads(manifest, buffers)


if __name__ == "__main__":
    main()
