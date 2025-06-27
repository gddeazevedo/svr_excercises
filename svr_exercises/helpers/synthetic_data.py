import numpy as np


def generate_linear_data(
        ang_coef: float,
        linear_coef: float,
        n_samples: int,
        with_noise: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
    '''
    Generate linear data with or without errors
    '''
    np.random.seed(42)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1) # shape = (100, 1)

    errors = 0

    if with_noise:
        errors = np.random.normal(0, 1, n_samples)

    y = ang_coef * X.squeeze() + linear_coef + errors

    return X, y
