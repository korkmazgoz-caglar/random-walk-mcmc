"""
Proposal distributions for RWMCMC
...
A proposal takes the initial and returns the proposed state. In this package, our proposals use symmetric distribution around the initial to create a new proposed state so q(x|x') = q(x'|x) and they cancel out each other during the acceptance criteria.
"""

import numpy as np
from numpy.typing import ArrayLike

def gaussian_random_walk(
    current_x: ArrayLike,
    step_size: float | ArrayLike = 1.0,
    rng: np.random.Generator | None = None

) -> np.ndarray:
    """
    Symmetric Gaussian random walk proposal distribution.
    x' = x + N(0, step_size)

    Parameters
    ----------
    current_x : np.ndarray
        The current state of the Markov chain.
    step_size : float or np.ndarray
        The standard deviation of the Gaussian distribution used for proposing new states.
    rng : np.random.Generator or None, default=None
        Random number generator.
    Returns
    -------
    np.ndarray
        The proposed new state.
    """
    
    if rng == None:
        rng = np.random.default_rng()
    
    current_x = np.asarray(current_x, dtype = float)
    epsilon = rng.normal(
        loc = 0.0,
        scale = step_size,
        size = current_x
    )
    proposal = current_x + epsilon
    
    return proposal