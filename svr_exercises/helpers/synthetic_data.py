import numpy as np
from typing import Callable


def generate_linear_data(
    ang_coef: float,
    linear_coef: float,
    n_samples: int,
    with_noise: bool = True,
    noise_std: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Gera dados lineares com ou sem ruído gaussiano controlável.

    Parâmetros:
    - ang_coef: coeficiente angular da reta
    - linear_coef: coeficiente linear da reta
    - n_samples: número de amostras
    - with_noise: se True, adiciona ruído
    - noise_std: intensidade (desvio padrão) do ruído

    Retorna:
    - X: array (n_samples, 1) com as entradas
    - y: array (n_samples,) com as saídas
    '''
    np.random.seed(42)  # para reprodutibilidade
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)

    if with_noise:
        noise = np.random.normal(loc=0.0, scale=noise_std, size=n_samples)
    else:
        noise = 0

    y = ang_coef * X.squeeze() + linear_coef + noise

    return X, y


def generate_nonlinear_data(
    func: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 100,
    noise_std: float = 0.1,
    with_noise: bool = True,
    x_min: float = 0.0,
    x_max: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Gera dados não lineares aplicando uma função a X e adicionando ruído opcional.

    Parâmetros:
    - func: função vetorizada que recebe np.ndarray e retorna np.ndarray (ex: np.sin)
    - n_samples: número de amostras
    - noise_std: desvio padrão do ruído
    - with_noise: se True, adiciona ruído
    - x_min: valor mínimo de X
    - x_max: valor máximo de X

    Retorna:
    - X: array (n_samples, 1)
    - y: array (n_samples,)
    '''
    np.random.seed(42)
    X = np.linspace(x_min, x_max, n_samples).reshape(-1, 1)

    if with_noise:
        noise = np.random.normal(loc=0.0, scale=noise_std, size=n_samples)
    else:
        noise = 0

    y = func(X).squeeze() + noise

    return X, y
