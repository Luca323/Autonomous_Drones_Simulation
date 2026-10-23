"""Python ctypes bridge to the native C++17 `fast_core_v2.dll` acceleration kernel."""

import ctypes
import subprocess
import sys
from pathlib import Path
from typing import Optional
import numpy as np

NATIVE_DIR = Path(__file__).parent
if sys.platform == "win32":
    LIB_NAME = "fast_core_v2.dll"
elif sys.platform == "darwin":
    LIB_NAME = "fast_core_v2.dylib"
else:
    LIB_NAME = "fast_core_v2.so"
DLL_PATH = NATIVE_DIR / LIB_NAME
CPP_PATH = NATIVE_DIR / "fast_core.cpp"

_LIB: Optional[ctypes.CDLL] = None


def _load_or_compile_lib() -> Optional[ctypes.CDLL]:
    global _LIB
    if _LIB is not None:
        return _LIB

    # Auto-compile if DLL does not exist or is older than fast_core.cpp
    if not DLL_PATH.exists() or (
        CPP_PATH.exists() and CPP_PATH.stat().st_mtime > DLL_PATH.stat().st_mtime
    ):
        compile_flags = [
            "-O3",
            "-ffast-math",
            "-shared",
        ]
        if sys.platform == "win32":
            compile_flags.extend(["-static-libgcc", "-static-libstdc++"])
        else:
            compile_flags.append("-fPIC")

        tmp_dll = NATIVE_DIR / f"{LIB_NAME}.tmp"
        old_dll = NATIVE_DIR / f"{LIB_NAME}.old"
        for compiler in [r"C:\msys64\ucrt64\bin\g++.exe", "g++", "clang++"]:
            try:
                subprocess.run(
                    [compiler, *compile_flags, "-o", str(tmp_dll), str(CPP_PATH)],
                    check=True,
                    capture_output=True,
                )
                if DLL_PATH.exists():
                    try:
                        if old_dll.exists():
                            old_dll.unlink()
                    except Exception:
                        pass
                    try:
                        DLL_PATH.rename(old_dll)
                    except Exception:
                        pass
                tmp_dll.replace(DLL_PATH)
                break
            except Exception:
                continue

    if not DLL_PATH.exists():
        return None

    try:
        lib = ctypes.CDLL(str(DLL_PATH))

        # int cpp_check_los(double x1, double y1, double x2, double y2, int n_obs, const double* obs_xyr, double margin)
        lib.cpp_check_los.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
        ]
        lib.cpp_check_los.restype = ctypes.c_int

        # void cpp_compute_matrices(...)
        lib.cpp_compute_matrices.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.cpp_compute_matrices.restype = None

        # int cpp_step_physics(...)
        lib.cpp_step_physics.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
        ]
        lib.cpp_step_physics.restype = ctypes.c_int

        # void cpp_compute_frontier_search(...)
        lib.cpp_compute_frontier_search.argtypes = [
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.cpp_compute_frontier_search.restype = None

        # void cpp_update_and_merge_grids(...)
        lib.cpp_update_and_merge_grids.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.cpp_update_and_merge_grids.restype = None

        _LIB = lib
        return _LIB
    except Exception:
        return None


def get_cpp_lib() -> Optional[ctypes.CDLL]:
    return _load_or_compile_lib()


def pack_obstacles_array(obstacles: list) -> np.ndarray:
    if not obstacles:
        return np.zeros((0, 3), dtype=np.float64)
    return np.ascontiguousarray(
        [[o.x, o.y, o.radius] for o in obstacles], dtype=np.float64
    )
