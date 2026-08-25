"""Stage B / Arm L1 environment readiness check for SpikeGPT on the local box.

Verifies, in order:
  0. GPU visible, torch+CUDA match, cupy importable (needed by the vendored
     spikingjelly `backend='cupy'` MultiStep LIF).
  1. The public SpikeGPT clone is present at external/SpikeGPT (re-clone with
     the command in READINESS.md if missing).
  2. Idempotently patches the clone for the reduced-scale dry run:
       - src/model.py: T_MAX 1024 -> 512 (compile-time; "TAKES LOTS OF VRAM")
       - src/model.py: LIF backend left at 'cupy' when cupy imports cleanly.
  3. Compiles the WKV CUDA extension with -DTmax=512 (needs nvcc + MSVC).
  4. Instantiates GPT at the Stage-B reduced config (L6/D384/T512, enwik8-class
     vocab) on CUDA, prints the parameter count and peak VRAM of one
     forward+backward at batch 8.

Run from experiments/bptt_derisk/stage_b/:
    python spikegpt_env_check.py [--skip-compile]

Exit code 0 = ready for the L1 dry run.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

STAGE_B_DIR = Path(__file__).resolve().parent
CLONE_DIR = STAGE_B_DIR / "external" / "SpikeGPT"

REDUCED_T_MAX = 512
N_LAYER = 6
N_EMBD = 384
CTX_LEN = 512
BATCH_SIZE = 8
ENWIK8_VOCAB = 205  # distinct bytes in enwik8 (char-level)

MSVC_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"),
    Path(r"C:\Program Files\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC"),
]
MSVC_VCVARS_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
]
NVCC_CANDIDATES = [
    Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
]


def bootstrap_msvc_env() -> bool:
    """Import the VS dev prompt environment (PATH/INCLUDE/LIB) into os.environ.

    torch.utils.cpp_extension on Windows locates MSVC with `where cl`, and cl
    in turn needs INCLUDE/LIB from vcvars64 to find the STL. Running from a
    plain shell neither is set, so build the environment ourselves once.

    Implementation note: `cmd /c call "<path>\\vcvars64.bat" && set` breaks on
    stock installs because cmd's compound-statement parser chokes on the
    parentheses in "Program Files (x86)". Routing through a one-shot temp
    batch file avoids the parser entirely.
    """
    if shutil.which("cl"):
        return True
    env_scripts = [Path(p) for p in MSVC_VCVARS_CANDIDATES]
    env_scripts += [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
    ]
    for script in env_scripts:
        if not script.is_file():
            continue
        with tempfile.NamedTemporaryFile(
            "w", suffix=".bat", delete=False, encoding="ascii"
        ) as bat:
            bat.write(f'@echo off\r\ncall "{script}" -arch=x64\r\nset\r\n')
            bat_path = bat.name
        try:
            proc = subprocess.run(
                ["cmd", "/c", bat_path],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[msvc] {script.name} exited {exc.returncode}; "
                  f"stderr: {(exc.stderr or '')[-500:]!r}")
            continue
        finally:
            try:
                os.unlink(bat_path)
            except OSError:
                pass
        for line in proc.stdout.splitlines():
            if "=" in line and not line.startswith("!"):
                key, _, value = line.partition("=")
                if key.upper() in {"PATH", "INCLUDE", "LIB", "LIBPATH"}:
                    os.environ[key] = value
                elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    os.environ.setdefault(key, value)
        if shutil.which("cl"):
            print(f"[msvc] bootstrapped dev env from {script}")
            return True
    print("[msvc] FAIL: no vcvars/VsDevCmd produced a cl.exe environment "
          "(VS Build Tools needs the 'Desktop development with C++' workload)")
    return False


def latest_hostx64(msvc_root: Path) -> Path | None:
    if not msvc_root.is_dir():
        return None
    versions = sorted((p for p in msvc_root.iterdir() if p.is_dir()), reverse=True)
    for vdir in versions:
        cl = vdir / "bin" / "Hostx64" / "x64" / "cl.exe"
        if cl.is_file():
            return cl.parent
    return None


def latest_nvcc(cuda_root: Path) -> tuple[Path, str] | None:
    if not cuda_root.is_dir():
        return None
    for vdir in sorted((p for p in cuda_root.iterdir() if p.is_dir()), reverse=True):
        nvcc = vdir / "bin" / "nvcc.exe"
        if nvcc.is_file():
            return nvcc.parent, vdir.name
    return None


def ensure_compiler_ready() -> bool:
    """Bootstrap the MSVC dev env (PATH/INCLUDE/LIB) and report toolchain.

    Deliberately does not hand-locate cl.exe for torch: cpp_extension uses
    `where cl`, which only works once vcvars64 has been imported. nvcc is
    found via CUDA_HOME / standard install paths by torch itself.
    """
    cl = latest_hostx64(MSVC_CANDIDATES[0])
    nvcc = latest_nvcc(NVCC_CANDIDATES[0])
    print(f"[toolchain] cl.exe on disk: {cl or 'NOT FOUND'}")
    print(f"[toolchain] nvcc:   {nvcc[0] if nvcc else 'NOT FOUND'}"
          + (f" (CUDA {nvcc[1]})" if nvcc else ""))
    ok = bootstrap_msvc_env()
    print(f"[toolchain] where cl -> {shutil.which('cl') or 'MISSING'}")
    return ok


def patch_clone() -> None:
    model_py = CLONE_DIR / "src" / "model.py"
    text = model_py.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r"^T_MAX = \d+",
        f"T_MAX = {REDUCED_T_MAX}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n == 0 and f"T_MAX = {REDUCED_T_MAX}" not in text:
        raise RuntimeError("could not locate T_MAX line in src/model.py")
    if new_text != text:
        model_py.write_text(new_text, encoding="utf-8")
        print(f"[patch] T_MAX -> {REDUCED_T_MAX} written")
    else:
        print(f"[patch] T_MAX already {REDUCED_T_MAX}")

    # Windows/nvcc>=12 fix: '--maxrregcount 60' as a single argv token is
    # rejected by modern nvcc ("Unknown option"); use -maxrregcount=60.
    new_text2 = new_text.replace("'--maxrregcount 60'", "'-maxrregcount=60'")
    # Same class of bug: '-Xptxas -O3' must be '-Xptxas=-O3'.
    new_text2 = new_text2.replace("'-Xptxas -O3'", "'-Xptxas=-O3'")
    if new_text2 != new_text:
        model_py.write_text(new_text2, encoding="utf-8")
        print("[patch] nvcc flag tokens normalized "
              "(-maxrregcount=60, -Xptxas=-O3)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-compile", action="store_true",
                    help="only verify imports/patch state, skip kernel build")
    args = ap.parse_args()

    # Vendored spikingjelly swallows import errors with logging.info — surface them.
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(name)s: %(message)s")

    import torch

    print(f"[torch] {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA unavailable"); return 2
    print(f"[gpu]  {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

    try:
        import cupy  # noqa: F401
        print(f"[cupy] {cupy.__version__} (vendored spikingjelly can use backend='cupy')")
    except Exception as exc:  # pragma: no cover
        print(f"[cupy] NOT IMPORTABLE ({exc}) -> patch backend='cupy'->'torch' before import")

    if not (CLONE_DIR / "src" / "model.py").is_file():
        print(f"FAIL: clone missing at {CLONE_DIR}\n"
              "      git clone --depth 1 https://github.com/ridgerchu/SpikeGPT "
              f"{CLONE_DIR}")
        return 2
    patch_clone()
    if args.skip_compile:
        return 0

    if not ensure_compiler_ready():
        return 2

    sys.path.insert(0, str(CLONE_DIR))
    os.chdir(CLONE_DIR)  # wkv load() uses relative source paths
    from src.model import GPT, GPTConfig  # triggers the wkv kernel compile

    # Diagnostic: why would check_backend see cupy as None?
    from src.spikingjelly.clock_driven import neuron as _sj_neuron
    print(f"[diag] spikingjelly neuron.cupy = {_sj_neuron.cupy}")

    cfg = GPTConfig(ENWIK8_VOCAB, CTX_LEN, model_type="RWKV",
                    n_layer=N_LAYER, n_embd=N_EMBD)
    model = GPT(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] RWKV-type SpikeGPT L{N_LAYER}/D{N_EMBD}/T{CTX_LEN}: "
          f"{n_params:,} params ({n_params/1e6:.2f}M)")

    idx = torch.randint(0, ENWIK8_VOCAB, (BATCH_SIZE, CTX_LEN), device="cuda")
    tgt = torch.randint(0, ENWIK8_VOCAB, (BATCH_SIZE, CTX_LEN), device="cuda")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    from src.spikingjelly.clock_driven import functional as sj_functional
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):  # warmup + measured step (first step includes JIT of cupy kernels)
        sj_functional.reset_net(model)  # LIF state must detach from prior graph
        opt.zero_grad(set_to_none=True)
        loss = model(idx, tgt)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    print(f"[step] loss={float(loss):.4f} (random-init sanity only)")
    print(f"[vram] peak allocated={peak:.0f} MiB  reserved={reserved:.0f} MiB  "
          f"at B{BATCH_SIZE}/T{CTX_LEN}")
    print("READY: reduced-scale SpikeGPT trains end-to-end on this GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
