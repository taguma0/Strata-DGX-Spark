"""tools/requant_dense_gguf.py - a copy of a dense GGUF whose natively served Q8_0 projections (and head) are re-quantized
to a smaller K- or legacy quant, so each decode window reads fewer dense bytes.

On a unified-memory box (DGX Spark: one LPDDR5X pool, ~230 GB/s) a decode window of the NVFP4 pack reads about 3.6 GiB of
Q8_0 dense matrices against about 2.7 GiB of experts; the window is bandwidth-bound, so every dense byte saved is time.
The layout is q8_dense_gguf.py's: the tensor-info block keeps its byte length (only each converted tensor's u32 type
and u64 offset change), the old data section stays, the new blocks are appended, every other tensor keeps its offset.

The quantization is ggml's own (ggml_quantize_chunk from a libggml-base built from llama.cpp - the K-quants have no
numpy encoder in gguf-py), no importance matrix. The source is the Q8_0 tensor dequantized to f32.

    python tools/requant_dense_gguf.py SRC.gguf DST.gguf --ggml-lib /path/libggml-base.so --type q5_k
        [--rule SUFFIX=TYPE ...]   per-tensor override, e.g. --rule output.weight=q6_k --rule attn_qkv.weight=q8_0

Then point the config's --native and --native-dense-gguf at DST.gguf. A row length that is not whole blocks of the
asked type keeps Q8_0.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402

_paths.add_gguf_py()
from gguf import GGUFReader, GGMLQuantizationType as QT  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

from q8_dense_gguf import is_target  # noqa: E402

TYPES = {"q4_0": QT.Q4_0, "q5_0": QT.Q5_0, "q5_1": QT.Q5_1, "q8_0": QT.Q8_0, "q4_k": QT.Q4_K, "q5_k": QT.Q5_K,
         "q6_k": QT.Q6_K, "iq4_xs": QT.IQ4_XS, "iq4_nl": QT.IQ4_NL}
BLOCK = {QT.Q4_0: (32, 18), QT.Q5_0: (32, 22), QT.Q5_1: (32, 24), QT.Q8_0: (32, 34), QT.Q4_K: (256, 144),
         QT.Q5_K: (256, 176), QT.Q6_K: (256, 210), QT.IQ4_XS: (256, 136), QT.IQ4_NL: (32, 18)}


def load_ggml(path: str):
    lib = ctypes.CDLL(path)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    return lib


def quantize(lib, f: np.ndarray, qt) -> np.ndarray:
    nrows, n = f.shape
    be, bb = BLOCK[qt]
    out = np.empty(nrows * (n // be) * bb, dtype=np.uint8)
    src = np.ascontiguousarray(f, dtype=np.float32)
    lib.ggml_quantize_init(int(qt))
    got = lib.ggml_quantize_chunk(int(qt), src.ctypes.data, out.ctypes.data, 0, nrows, n, None)
    assert got == out.nbytes, (got, out.nbytes)
    return out


def pick(name: str, default, rules) -> object:
    for suffix, qt in rules:
        if name.endswith(suffix):
            return qt
    return default


def jobs_of(r, default, rules) -> list:
    data_start, mm, jobs = r.data_offset, r.data, []
    for t in r.tensors:
        if not is_target(t.name) or int(t.tensor_type) != int(QT.Q8_0):
            continue
        qt = pick(t.name, default, rules)
        if qt == QT.Q8_0:
            continue
        if int(t.shape[0]) % BLOCK[qt][0]:
            print(f"  {t.name}: rows of {int(t.shape[0])} are not whole {qt.name} blocks, kept Q8_0")
            continue
        p = t.field.offset                                   # the tensor-info entry: name, n_dims, dims, type, offset
        nlen = struct.unpack_from("<Q", mm, p)[0]; p += 8
        assert bytes(mm[p:p + nlen]).decode() == t.name; p += nlen
        ndim = struct.unpack_from("<I", mm, p)[0]; p += 4 + 8 * ndim
        ttype, toff = struct.unpack_from("<IQ", mm, p)
        assert ttype == int(t.tensor_type) and data_start + toff == t.data_offset, t.name
        jobs.append((t, p, qt))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--ggml-lib", required=True, help="libggml-base.so (or .dylib) built from llama.cpp")
    ap.add_argument("--type", default="q5_k", choices=sorted(TYPES))
    ap.add_argument("--rule", action="append", default=[], help="SUFFIX=TYPE, first match wins")
    a = ap.parse_args()
    if a.dst.resolve() == a.src.resolve():
        sys.exit("the output must not be the source")
    rules = []
    for rule in a.rule:
        suffix, _, ty = rule.partition("=")
        rules.append((suffix, TYPES[ty.lower()]))
    lib = load_ggml(a.ggml_lib)
    r = GGUFReader(a.src)
    jobs = jobs_of(r, TYPES[a.type], rules)
    t0 = time.time()
    blobs, before, worst = [], 0, 0.0
    for t, p, qt in jobs:
        ne0, ne1 = int(t.shape[0]), int(t.shape[1])
        f = np.asarray(dequantize(np.asarray(t.data), t.tensor_type), dtype=np.float32).reshape(ne1, ne0)
        q = quantize(lib, f, qt)
        back = np.asarray(dequantize(q, qt), dtype=np.float32).reshape(f.shape)
        rel = float(np.sqrt(np.mean((back - f) ** 2)) / max(float(np.sqrt(np.mean(f ** 2))), 1e-30))
        worst = max(worst, rel)
        before += int(t.n_bytes)
        blobs.append((t, p, qt, q))
    after = sum(b[3].nbytes for b in blobs)
    print(f"{len(blobs)} tensors re-quantized in {time.time() - t0:.0f} s: {before / 2**30:.2f} -> {after / 2**30:.2f} GiB, "
          f"worst relative RMS error {worst:.3%}", flush=True)

    align, data_start, src_size = r.alignment, r.data_offset, os.path.getsize(a.src)
    tmp = a.dst.with_name(a.dst.name + ".part")
    shutil.copyfile(a.src, tmp)
    with open(tmp, "r+b") as fh:
        end = src_size
        for t, p, qt, blob in blobs:
            at = (end + align - 1) // align * align
            fh.seek(end); fh.write(b"\0" * (at - end))
            fh.seek(at); fh.write(blob.tobytes())
            end = at + blob.nbytes
            fh.seek(p); fh.write(struct.pack("<IQ", int(qt), at - data_start))
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, a.dst)

    v = GGUFReader(a.dst)                                    # every other tensor where it was, every new one as written
    conv = {t.name: (qt, blob) for t, _, qt, blob in blobs}
    orig = {t.name: t for t in r.tensors}
    assert len(v.tensors) == len(r.tensors) and v.data_offset == data_start
    for t in v.tensors:
        if t.name in conv:
            qt, blob = conv[t.name]
            assert t.tensor_type == qt and np.array_equal(np.asarray(t.data).view(np.uint8).reshape(-1), blob), t.name
        else:
            assert t.tensor_type == orig[t.name].tensor_type and t.data_offset == orig[t.name].data_offset, t.name
    print(f"wrote {a.dst} ({os.path.getsize(a.dst) / 2**30:.2f} GiB), offsets checked", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
