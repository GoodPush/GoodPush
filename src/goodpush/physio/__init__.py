"""Physiological state-space model of one athlete, fitted as a Bayesian inversion (JAX / NumPyro).

Separate from the rest of goodpush: numpy, scipy, jax (+ numpyro for sampling) only, inputs are plain
arrays (`PhysioData`). goodpush builds those in models/physio.py.
"""
import os
from importlib.metadata import version

# CPU "devices" for parallel NUTS chains: must be set before JAX starts its backend
CHAINS = min(4, os.cpu_count() or 1)
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={CHAINS}").strip()
# one thread per device: the parallelism is the chains (and numba's own kernels, NUMBA_NUM_THREADS); the model's
# operations are small, so threads inside a chain would cost more in overhead than they give, and nest
if "xla_cpu_multi_thread_eigen" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1").strip()
# XLA's older CPU runtime: the model's day-by-day scan runs ~1.5x faster on it (jaxlib 0.4.38, 2026-10-27).
# Newer jaxlibs drop the flag (an unknown flag is fatal), so only up to 0.4
if version("jaxlib").startswith("0.4.") and "xla_cpu_use_thunk_runtime" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_cpu_use_thunk_runtime=false").strip()

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from .curve import PowerCurve                                            # noqa: E402
from .fit import MapResult, PhysioFit                                    # noqa: E402
from .model import DOSES, KINDS, QUANTITIES, STATES, PhysioData, PhysioModel, Source  # noqa: E402

__all__ = ["PowerCurve", "PhysioData", "PhysioModel", "PhysioFit", "MapResult", "Source",
           "STATES", "DOSES", "QUANTITIES", "KINDS"]
