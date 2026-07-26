"""Gera o dataAug do Kyoto e regenera os CSVs Kyoto_autoencoder_*.

Os CSVs legados referenciam imagens aumentadas em {Kyoto}/dataAug que não
acompanham o repositório. Este script recria a pasta a partir dos PNGs crus
(split 32/20/8 idêntico ao de dataset.py::create_Kyoto_csv, aumentando apenas
as imagens de treino) e reescreve os 3 CSVs com paths locais.

Uso:
    python scripts/make_kyoto_dataaug.py [--force]
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import paths

CSV_DIR, KYOTO_DIR = paths.CSV_DIR, paths.KYOTO_DIR

SEED = 42
N_TRAIN, N_VAL, N_TEST = 32, 20, 8


def augmentations(image: Image.Image, rng: random.Random):
    width, height = image.size

    def crop(fraction):
        crop_w, crop_h = int(width * fraction), int(height * fraction)
        left = rng.randint(0, width - crop_w)
        top = rng.randint(0, height - crop_h)
        return image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height))

    return [
        image.transpose(Image.FLIP_LEFT_RIGHT),
        image.transpose(Image.FLIP_TOP_BOTTOM),
        image.transpose(Image.ROTATE_90),
        image.transpose(Image.ROTATE_180),
        image.transpose(Image.ROTATE_270),
        image.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT),
        crop(0.8),
        crop(0.6),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Regera o dataAug mesmo que a pasta já exista")
    args = parser.parse_args()

    if not KYOTO_DIR.is_dir():
        sys.exit(f"Dataset Kyoto não encontrado em {KYOTO_DIR} "
                 "(defina KYOTO_DIR ou DATASETS_ROOT)")

    raw_images = sorted(
        p for p in KYOTO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if len(raw_images) < N_TRAIN + N_TEST:
        sys.exit(f"Apenas {len(raw_images)} imagens em {KYOTO_DIR}; "
                 f"esperado ao menos {N_TRAIN + N_TEST}")

    train_raw = raw_images[:N_TRAIN]
    validation = raw_images[N_TRAIN:N_TRAIN + N_VAL]
    test = raw_images[-N_TEST:]

    dataaug_dir = KYOTO_DIR / "dataAug"
    existing = sorted(dataaug_dir.glob("*.jpg")) if dataaug_dir.is_dir() else []

    if existing and not args.force:
        print(f"dataAug já existe com {len(existing)} imagens; usando como está "
              "(--force para regerar).")
        aug_paths = existing
    else:
        dataaug_dir.mkdir(exist_ok=True)
        for old in existing:
            old.unlink()
        rng = random.Random(SEED)
        aug_paths = []
        for src in train_raw:
            with Image.open(src) as img:
                image = img.convert("RGB")
                for k, augmented in enumerate(augmentations(image, rng)):
                    out = dataaug_dir / f"kyoto_{src.stem}_{k}.jpg"
                    augmented.save(out, quality=95)
                    aug_paths.append(out)
        print(f"{len(aug_paths)} imagens aumentadas geradas em {dataaug_dir}")

    out_dir = CSV_DIR / "Kyoto"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_paths = [str(p) for p in train_raw] + sorted(str(p) for p in aug_paths)
    splits = {
        "train": train_paths,
        "validation": [str(p) for p in validation],
        "test": [str(p) for p in test],
    }
    for split, paths in splits.items():
        csv_path = out_dir / f"Kyoto_autoencoder_{split}.csv"
        pd.DataFrame({"path_image": paths}).to_csv(csv_path, index=False)
        print(f"{csv_path}: {len(paths)} linhas")


if __name__ == "__main__":
    main()
