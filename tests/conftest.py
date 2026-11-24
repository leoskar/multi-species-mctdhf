"""Test-session configuration.

The one thing that matters here is ORDER: the refactor pins must run before
anything else in the process.

They compare propagated trajectories against recordings made on the
pre-refactor tree. On CPU that comparison is exact on the recording
machine type (see ``tests/test_refactor_pins.py``). On GPU
it is exact only in a clean process: XLA's compiled-code state, its
autotuning choices and the allocator's arena are process-global, so running
~280 other tests first changes how these same computations are fused and
moves the result. Measured on an RTX 2070 SUPER, the imaginary-time
``diagonalize`` cases reproduce the recording BIT-FOR-BIT in a fresh process
(three independent runs), and drift to 2.5e-12 on the orbitals — about 1e4
epsilons, via the amplification of the regularised density inverse — when
they run at the end of the full suite.

Loosening the tolerance to survive that would throw away the pin's whole
value. Running the pins first keeps them sharp, costs nothing, and is
honest about what they measure.
"""

import os


_PIN_MODULE = "test_refactor_pins"


def pytest_collection_modifyitems(items):
    """Move the refactor pins to the front of the session."""
    pins = [it for it in items if _PIN_MODULE in str(it.fspath)]
    if not pins:
        return
    rest = [it for it in items if _PIN_MODULE not in str(it.fspath)]
    items[:] = pins + rest
