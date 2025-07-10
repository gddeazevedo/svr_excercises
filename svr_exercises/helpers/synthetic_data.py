import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable


def generate_linear_data(
    ang_coef: float,
    linear_coef: float,
    n_samples: int,
    with_noise: bool = False,
    noise_std: float = 1.0,
    with_outliers: bool = False,
    n_outliers: int = 0,
    outlier_offset: float = 0
) -> Tuple[NDArray, NDArray]:
    """
    Gera dados lineares com ou sem ruído e outliers.

    Parâmetros:
    - ang_coef: coeficiente angular (inclinação)
    - linear_coef: coeficiente linear (intercepto)
    - n_samples: número de amostras normais
    - with_noise: se True, adiciona ruído gaussiano
    - noise_std: desvio padrão do ruído
    - with_outliers: se True, insere outliers
    - n_outliers: número de outliers a inserir
    - offset: magnitude do deslocamento dos outliers

    Retorna:
    - X: array (n_total, 1) com as entradas
    - y: array (n_total,) com as saídas
    """
    np.random.seed(42)

    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=n_samples) if with_noise else 0
    y = ang_coef * X.squeeze() + linear_coef + noise

    if with_outliers and n_outliers > 0:
        X_outliers, y_outliers = generate_outliers_for_linear_data(
            ang_coef=ang_coef,
            linear_coef=linear_coef,
            offset=outlier_offset,
            n_outliers=n_outliers,
            noise_std=noise_std
        )
        X = np.concatenate([X, X_outliers])
        y = np.concatenate([y, y_outliers])

    return X, y


def generate_outliers_for_linear_data(
    ang_coef: float,
    linear_coef: float,
    offset: float,
    n_outliers: int,
    noise_std: float = 1.0
) -> Tuple[NDArray, NDArray]:
    """
    Gera outliers distribuídos simetricamente acima e abaixo da reta.

    Parâmetros:
    - ang_coef: coeficiente angular da reta base
    - offset: deslocamento dos outliers
    - n_outliers: número total de outliers (deve ser par)
    - noise_std: desvio padrão do ruído nos outliers

    Retorna:
    - X_outliers: array (n_outliers, 1) com os X dos outliers
    - y_outliers: array (n_outliers,) com os Y dos outliers
    """
    X_outliers = np.random.uniform(2, 8, size=n_outliers).reshape(-1, 1)
    y_base = ang_coef * X_outliers.ravel() + linear_coef
    offsets = np.array([offset] * (n_outliers // 2) + [-offset] * (n_outliers // 2))
    np.random.shuffle(offsets)
    y_outliers = y_base + offsets + np.random.normal(0, noise_std, size=n_outliers)

    return X_outliers, y_outliers

def generate_nonlinear_data(
    func: Callable[[NDArray], NDArray],
    n_samples: int = 100,
    noise_std: float = 1.0,
    with_noise: bool = True,
    x_min: float = 0.0,
    x_max: float = 10.0
) -> Tuple[NDArray, NDArray]:
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
