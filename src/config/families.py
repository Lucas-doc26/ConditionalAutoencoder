"""Registro central das famílias de autoencoders e do protocolo de avaliação.

Fonte única de verdade para os pipelines *_families: quais classes compõem
cada família, em quais bases os autoencoders são pré-treinados e quais
datasets treinam/avaliam os classificadores (mesmo protocolo do legado
Modelos/Modelo_Kyoto-*).
"""

from dataclasses import dataclass, field

from src.models.autoencoders import (
    Autoencoder0, Autoencoder1, Autoencoder2, Autoencoder3, Autoencoder4,
    Autoencoder5, Autoencoder6, Autoencoder7, Autoencoder8, Autoencoder9,
    Encoder0, Encoder1, Encoder2, Encoder3, Encoder4,
    Encoder5, Encoder6, Encoder7, Encoder8, Encoder9,
)
from src.models.autoencoders_with_skip_connections import (
    SkipAutoencoder0, SkipAutoencoder1, SkipAutoencoder2, SkipAutoencoder3,
    SkipAutoencoder4, SkipAutoencoder5, SkipAutoencoder6, SkipAutoencoder7,
    SkipAutoencoder8, SkipAutoencoder9,
    SkipEncoder0, SkipEncoder1, SkipEncoder2, SkipEncoder3, SkipEncoder4,
    SkipEncoder5, SkipEncoder6, SkipEncoder7, SkipEncoder8, SkipEncoder9,
)
from src.models.variational_autoencoder import (
    VariationalAutoencoder0, VariationalAutoencoder1, VariationalAutoencoder2,
    VariationalAutoencoder3, VariationalAutoencoder4, VariationalAutoencoder5,
    VariationalAutoencoder6, VariationalAutoencoder7, VariationalAutoencoder8,
    VariationalAutoencoder9,
    VariationalEncoder0, VariationalEncoder1, VariationalEncoder2,
    VariationalEncoder3, VariationalEncoder4, VariationalEncoder5,
    VariationalEncoder6, VariationalEncoder7, VariationalEncoder8,
    VariationalEncoder9,
)
from src.models.joint_autoencoder import (
    JointAutoencoder0, JointAutoencoder1,
    JointEncoder0, JointEncoder1,
    JointSkipAutoencoder0, JointSkipAutoencoder1,
    JointSkipEncoder0, JointSkipEncoder1,
)


@dataclass(frozen=True)
class Family:
    key: str            # nome curto usado na CLI (ae, skip, vae, joint, jointskip)
    prefix: str         # prefixo das pastas em Modelos/ (ex.: Modelo_Skip)
    autoencoders: list = field(default_factory=list)
    encoders: list = field(default_factory=list)

    @property
    def n_models(self):
        return len(self.autoencoders)


FAMILIES = {
    "ae": Family(
        key="ae", prefix="Modelo_AE",
        autoencoders=[
            Autoencoder0, Autoencoder1, Autoencoder2, Autoencoder3, Autoencoder4,
            Autoencoder5, Autoencoder6, Autoencoder7, Autoencoder8, Autoencoder9,
        ],
        encoders=[
            Encoder0, Encoder1, Encoder2, Encoder3, Encoder4,
            Encoder5, Encoder6, Encoder7, Encoder8, Encoder9,
        ],
    ),
    "skip": Family(
        key="skip", prefix="Modelo_Skip",
        autoencoders=[
            SkipAutoencoder0, SkipAutoencoder1, SkipAutoencoder2, SkipAutoencoder3,
            SkipAutoencoder4, SkipAutoencoder5, SkipAutoencoder6, SkipAutoencoder7,
            SkipAutoencoder8, SkipAutoencoder9,
        ],
        encoders=[
            SkipEncoder0, SkipEncoder1, SkipEncoder2, SkipEncoder3, SkipEncoder4,
            SkipEncoder5, SkipEncoder6, SkipEncoder7, SkipEncoder8, SkipEncoder9,
        ],
    ),
    "vae": Family(
        key="vae", prefix="Modelo_VAE",
        autoencoders=[
            VariationalAutoencoder0, VariationalAutoencoder1, VariationalAutoencoder2,
            VariationalAutoencoder3, VariationalAutoencoder4, VariationalAutoencoder5,
            VariationalAutoencoder6, VariationalAutoencoder7, VariationalAutoencoder8,
            VariationalAutoencoder9,
        ],
        encoders=[
            VariationalEncoder0, VariationalEncoder1, VariationalEncoder2,
            VariationalEncoder3, VariationalEncoder4, VariationalEncoder5,
            VariationalEncoder6, VariationalEncoder7, VariationalEncoder8,
            VariationalEncoder9,
        ],
    ),
    "joint": Family(
        key="joint", prefix="Modelo_Joint",
        autoencoders=[JointAutoencoder0, JointAutoencoder1],
        encoders=[JointEncoder0, JointEncoder1],
    ),
    "jointskip": Family(
        key="jointskip", prefix="Modelo_JointSkip",
        autoencoders=[JointSkipAutoencoder0, JointSkipAutoencoder1],
        encoders=[JointSkipEncoder0, JointSkipEncoder1],
    ),
}

# Bases de pré-treino dos autoencoders (um conjunto de pesos por base).
BASES = ["CNR", "PKLot", "Kyoto"]

# Datasets de treino dos classificadores por base do encoder (protocolo legado).
CAMERAS = [f"camera{i}" for i in range(1, 10)]
PKLOT_SETS = ["PUC", "UFPR04", "UFPR05"]

BASE_TO_TRAINS = {
    "CNR": PKLOT_SETS,
    "PKLot": CAMERAS,
    "Kyoto": PKLOT_SETS + CAMERAS,
}

# Todos os classificadores são avaliados nos 12 datasets.
DATASETS_TEST = PKLOT_SETS + CAMERAS

# Tamanhos de batch de treino dos classificadores (CSV/{train}/batches/batch-N.csv).
BATCH_SIZES_CSV = [64, 128, 256, 512, 1024]


def resolve_families(arg: str):
    """Converte "ae,skip" ou "all" na lista de Family correspondente."""
    if arg == "all":
        return list(FAMILIES.values())
    families = []
    for key in arg.split(","):
        key = key.strip().lower()
        if key not in FAMILIES:
            raise ValueError(
                f"Família inválida: {key}. Use all ou {'|'.join(FAMILIES)}"
            )
        families.append(FAMILIES[key])
    return families


def resolve_indices(arg: str, family: Family):
    """Converte "0-9", "0,3,7" ou "all" nos índices válidos da família."""
    if arg == "all":
        return list(range(family.n_models))
    indices = []
    for part in arg.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return [i for i in indices if i < family.n_models]
