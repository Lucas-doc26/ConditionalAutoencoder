"""Helpers de path espelhando a estrutura legada de Modelos/Modelo_Kyoto-*.

Atenção às idiossincrasias do legado, que a fusão espera:
Pesos/ e Precisao/ usam "Treinado_em_"; Resultados/ usa "Treinados_em_".
"""

from src.config.paths import MODELOS_DIR


def model_dir(prefix, i):
    return MODELOS_DIR / f"{prefix}-{i}"


def base_weights_path(prefix, i, base, part=None):
    """part: None (autoencoder completo), "encoder" ou "decoder"."""
    suffix = f"_{part}" if part else ""
    return (model_dir(prefix, i) / "Modelo-Base" / "Pesos"
            / f"{prefix}-{i}_Base-{base}{suffix}.pth")


def classifier_weights_path(prefix, i, base, train, batch):
    return (model_dir(prefix, i) / f"Classificador-{base}" / "Pesos"
            / f"Treinado_em_{train}" / f"Classificador_{prefix}-{i}_batches-{batch}.pth")


def precisao_path(prefix, i, base, train, test):
    return (model_dir(prefix, i) / f"Classificador-{base}" / "Precisao"
            / f"Treinado_em_{train}" / f"precisao-{test}.txt")


def npy_path(prefix, i, base, train, test, batch):
    return (model_dir(prefix, i) / f"Classificador-{base}" / "Resultados"
            / f"Treinados_em_{train}" / test / f"batches-{batch}.npy")
