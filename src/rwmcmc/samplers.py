import numpy as np

def random_walk_metropolis_hastings_1d(
    target_log_pdf,
    x0: int,
    n_samples: int,
    step_size: float = 1.0,
    rng: np.random.Generator | None = None,


) -> np.ndarray:

    """Docstring should be added for parameter descriptions and return values, as well as a brief explanation of the algorithm and its purpose.

    The random walk Metropolis-Hastings algorithm is a Markov Chain Monte Carlo (MCMC) method used to sample from a target distribution when direct sampling is difficult. It constructs a Markov chain that has the target distribution as its stationary distribution. The algorithm iteratively proposes new samples based on the current sample and accepts or rejects them based on the acceptance ratio, which is calculated using the target distribution's log probability density function. The step size parameter controls the scale of the random walk, and the random number generator (rng) can be used to ensure reproducibility of the sampling process. The function returns an array of samples drawn from the target distribution and a boolean array indicating which proposals were accepted. This implementation assumes a symmetric proposal distribution, which simplifies the acceptance ratio calculation.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    samples = np.zeros(n_samples)
    samples[0] = x0
    
    accepted = np.zeros(n_samples, dtype=bool) # To keep track of accepted proposals, it is not necessary for the algorithm to work, but it can be useful for diagnostics and analysis of the sampling process.

    log_p_current = target_log_pdf(x0)
    # Building the Markov chain
    for i in range(1, n_samples):
        current_x = samples[i - 1]
        epsilon = rng.normal(loc=0.0, scale=step_size) # Draws a random number from Gaussian distribution with mean 0 and standard deviation step_size, it could be any symmetric distribution, but Gaussian is common for random walk proposals. Which is interesting as our target distribution is also Gaussian, but it is not a requirement for the algorithm to work.
        proposed_x = current_x + epsilon
        
        log_p_proposed = target_log_pdf(proposed_x)
        log_acceptance_ratio_alpha = log_p_proposed - log_p_current # Log of acceptance ratio, we can use log probabilities to avoid numerical underflow issues when dealing with very small probabilities.
        
        # Compare the log of acceptance ratio if it is greater than zero, we accept the proposal, otherwise we compare with a random number from uniform distribution to decide whether to accept or reject the proposal.
        if log_acceptance_ratio_alpha >= 0 or rng.uniform() < np.exp(log_acceptance_ratio_alpha):
            samples[i] = proposed_x
            accepted[i] = True
            log_p_current = log_p_proposed # Update the current log probability to the proposed one, as we have accepted the proposal.
        else:
            samples[i] = current_x
            accepted[i] = False


    return samples, accepted
            
"""

python -c "
import numpy as np
from rwmcmc.targets import gaussian_1d_log_pdf
from rwmcmc.samplers import random_walk_metropolis_hastings

samples, accepted = random_walk_metropolis_hastings(
    target_log_pdf=gaussian_1d_log_pdf,
    x0=0.0,
    n_samples=10000,
    step_size=2.0,
)
print(f'Acceptance rate: {accepted.mean():.3f}')
print(f'Sample mean: {samples.mean():.4f}')
print(f'Sample std:  {samples.std():.4f}')
"

"""