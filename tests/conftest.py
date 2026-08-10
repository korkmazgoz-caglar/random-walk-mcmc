"""Shared test setup.

Force a non-interactive matplotlib backend so the plotting tests behave the
same way on a laptop and on a CI runner without a display.
"""

import matplotlib

matplotlib.use("Agg")
