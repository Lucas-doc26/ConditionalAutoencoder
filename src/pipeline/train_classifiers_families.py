"""Classificadores (encoder congelado + MLP) para todas as famílias, espelhando Modelos/.

Para cada família × índice × base do encoder × dataset de treino
(BASE_TO_TRAINS) × batch (64..1024), treina o classificador com
CSV/{train}/batches/batch-{N}.csv e avalia nos 12 datasets de teste, salvando:

  Modelos/{prefix}-{i}/Classificador-{base}/
      Pesos/Treinado_em_{train}/Classificador_{prefix}-{i}_batches-{N}.pth
      Precisao/Treinado_em_{train}/precisao-{teste}.txt   (uma linha por batch)
      Resultados/Treinados_em_{train}/{teste}/batches-{N}.npy  (probs float32, N×2)

Protocolo de teste (idêntico ao legado Modelo_Kyoto-*):
  teste == treino -> CSV/{teste}/{teste}_test.csv; senão -> CSV/{teste}/{teste}.csv.
Os .npy seguem exatamente a ordem de linhas desses CSVs (shuffle=False).

Idempotente: pula (train, batch) cujos 12 .npy já existem; os precisao-*.txt são
reconstruídos a partir dos .npy salvos. MLflow: local (./mlruns) sem --mlflow.

Uso:
    python -m src.pipeline.train_classifiers_families -e 10 --families all
"""

import argparse
import os

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, split_jobs
from src.config.families import (
    BASES, BASE_TO_TRAINS, BATCH_SIZES_CSV, DATASETS_TEST,
    FAMILIES, resolve_families, resolve_indices,
)
from src.config.paths import CSV_DIR
from src.models import Classifier
from src.utils.datasets import CustomImageDataset
from src.utils.model_paths import (
    base_weights_path, classifier_weights_path, npy_path, precisao_path,
)
from src.utils.transform import return_transform_64

NUM_WORKS = 2
PERSISTENT_WORKERS = True

torch.backends.cudnn.benchmark = True


def unique_devices():
    return list(dict.fromkeys(Config().DEVICES))


def test_csv_for(test, train):
    """CSV de teste do protocolo legado: split _test quando teste == treino."""
    name = f"{test}_test.csv" if test == train else f"{test}.csv"
    return CSV_DIR / test / name


def make_loader(csv_path, transform, batch_size, pin_memory, shuffle):
    dataset = CustomImageDataset(str(csv_path), transform=transform, autoencoder=False)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKS,
        pin_memory=pin_memory, persistent_workers=PERSISTENT_WORKERS,
    )


def batch_done(family, index, base, train, batch):
    return all(
        npy_path(family.prefix, index, base, train, test, batch).is_file()
        for test in DATASETS_TEST
    )


def load_encoder(family, index, base, device):
    encoder = family.encoders[index]().to(device)
    weights = base_weights_path(family.prefix, index, base, "encoder")
    encoder.load_state_dict(torch.load(weights, map_location=device))
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder


def evaluate_and_save(model, family, index, base, train, batch, device, transform,
                      batch_size, pin_memory, use_amp):
    """Avalia nos 12 testes, salva os .npy alinhados e retorna {teste: acc}."""
    accs = {}
    model.eval()
    for test in DATASETS_TEST:
        csv_path = test_csv_for(test, train)
        n_rows = len(pd.read_csv(csv_path))
        loader = make_loader(csv_path, transform, batch_size, pin_memory, shuffle=False)

        all_probs, all_ids, y_true = [], [], []
        with torch.no_grad():
            for x, y, idx in tqdm(loader, desc=f"test {test}", leave=False):
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(x)
                probs = torch.softmax(out.float(), dim=1)
                all_probs.append(probs.cpu().numpy().astype(np.float32))
                all_ids.extend(idx.numpy().tolist())
                y_true.extend(y.numpy().tolist())

        probs = np.concatenate(all_probs)
        ids = np.asarray(all_ids)

        # Garantias duras de alinhamento npy <-> CSV (a fusão depende disso).
        assert probs.shape == (n_rows, 2), f"{test}: {probs.shape} != ({n_rows}, 2)"
        assert (ids == np.arange(n_rows)).all(), f"{test}: ids fora de ordem"

        out_path = npy_path(family.prefix, index, base, train, test, batch)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, probs)

        accs[test] = accuracy_score(y_true, probs.argmax(axis=1))
        del loader
        if use_amp:
            torch.cuda.empty_cache()
    return accs


def recompute_acc_from_npy(family, index, base, train, batch, test):
    probs = np.load(npy_path(family.prefix, index, base, train, test, batch))
    y_true = pd.read_csv(test_csv_for(test, train))["class"].values
    assert len(y_true) == len(probs), f"{test}: npy desalinhado do CSV"
    return accuracy_score(y_true, probs.argmax(axis=1))


def write_precisao(family, index, base, train, accs_by_batch):
    """Reescreve precisao-{teste}.txt com uma linha por batch (ordem 64..1024)."""
    for test in DATASETS_TEST:
        path = precisao_path(family.prefix, index, base, train, test)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{accs_by_batch[batch][test]:.6f}" for batch in BATCH_SIZES_CSV
                 if batch in accs_by_batch]
        path.write_text("\n".join(lines) + "\n")


def train_job(device, family, index, base, train, num_epochs, batch_size=32, lr=1e-3,
              resume=True):
    """Treina os 5 classificadores (um por batch CSV) de um (família, índice, base, train)."""
    transform = return_transform_64()
    use_amp = str(device).startswith("cuda")
    pin_memory = use_amp

    encoder = load_encoder(family, index, base, device)
    model_name = f"Classificador_{family.prefix}-{index}"
    mlflow.set_experiment(f"Classifier64_{family.prefix}")

    accs_by_batch = {}

    for batch in BATCH_SIZES_CSV:
        if resume and batch_done(family, index, base, train, batch):
            accs_by_batch[batch] = {
                test: recompute_acc_from_npy(family, index, base, train, batch, test)
                for test in DATASETS_TEST
            }
            print(f"[resume] {model_name} Base-{base} {train} batch-{batch} já existe")
            continue

        model = Classifier(encoder, latent_dim=encoder.latent_dim, num_classes=2).to(device)

        train_loader = make_loader(
            CSV_DIR / train / "batches" / f"batch-{batch}.csv",
            transform, batch_size, pin_memory, shuffle=True,
        )
        val_loader = make_loader(
            CSV_DIR / train / f"{train}_validation.csv",
            transform, batch_size, pin_memory, shuffle=False,
        )

        with mlflow.start_run(run_name=f"{model_name}_Base-{base}_{train}_{batch}"):
            mlflow.log_params({
                "model": model_name,
                "family": family.key,
                "encoder_dataset": base,
                "dataset_classifier": train,
                "n_images_to_train": batch,
                "epochs": num_epochs,
                "lr": lr,
                "loss": "CrossEntropy",
                "input_shape": "3x64x64",
            })

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            for epoch in range(num_epochs):
                model.train()
                encoder.eval()
                train_loss, correct, total = 0.0, 0, 0
                for x, y, _ in train_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out = model(x)
                        loss = criterion(out, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss += loss.item()
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)

                model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for x, y, _ in val_loader:
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            out = model(x)
                            loss = criterion(out, y)
                        val_loss += loss.item()
                        val_correct += (out.argmax(1) == y).sum().item()
                        val_total += y.size(0)

                mlflow.log_metric("train_loss", train_loss / max(len(train_loader), 1), step=epoch)
                mlflow.log_metric("train_acc", correct / max(total, 1), step=epoch)
                mlflow.log_metric("val_loss", val_loss / max(len(val_loader), 1), step=epoch)
                mlflow.log_metric("val_acc", val_correct / max(val_total, 1), step=epoch)

            accs = evaluate_and_save(
                model, family, index, base, train, batch, device, transform,
                batch_size, pin_memory, use_amp,
            )
            accs_by_batch[batch] = accs
            for test, acc in accs.items():
                mlflow.log_metric(f"test_acc-{test}", acc)

            weights = classifier_weights_path(family.prefix, index, base, train, batch)
            weights.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), weights)

        del train_loader, val_loader, model
        if use_amp:
            torch.cuda.empty_cache()
        print(f"✓ {model_name} Base-{base} {train} batch-{batch}")

    write_precisao(family, index, base, train, accs_by_batch)


def worker(rank, jobs_split, devices, epochs, use_mlflow, resume):
    device = devices[rank % len(devices)]
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)
    if use_mlflow:
        mlflow.set_tracking_uri(Config().IP_LOCAL)

    for family_key, index, base, train in jobs_split[rank]:
        family = FAMILIES[family_key]
        train_job(device, family, index, base, train, num_epochs=epochs, resume=resume)


def dataset_available(train):
    csv_path = CSV_DIR / train / "batches" / "batch-64.csv"
    if not csv_path.is_file():
        return False, f"CSV inexistente: {csv_path}"
    with open(csv_path) as f:
        f.readline()
        first = f.readline().strip().split(",")[0]
    if not first or not os.path.exists(first):
        return False, f"imagens indisponíveis (ex.: {first})"
    return True, ""


def build_jobs(args):
    jobs, skipped = [], []
    bases = [b.strip() for b in args.bases.split(",")]
    trains_filter = ([t.strip() for t in args.trains.split(",")]
                     if args.trains != "all" else None)
    availability = {}

    for base in bases:
        if base not in BASES:
            raise ValueError(f"Base inválida: {base}. Use {BASES}")
        for family in resolve_families(args.families):
            for index in resolve_indices(args.indices, family):
                if not base_weights_path(family.prefix, index, base, "encoder").is_file():
                    print(f"[pulando] {family.prefix}-{index} Base-{base}: "
                          "encoder .pth não encontrado (rode a etapa de autoencoders)")
                    continue
                for train in BASE_TO_TRAINS[base]:
                    if trains_filter and train not in trains_filter:
                        continue
                    if train not in availability:
                        availability[train] = dataset_available(train)
                    ok, reason = availability[train]
                    if not ok:
                        continue
                    if not args.no_resume and all(
                        batch_done(family, index, base, train, b) for b in BATCH_SIZES_CSV
                    ):
                        skipped.append((family.prefix, index, base, train))
                        continue
                    jobs.append((family.key, index, base, train))

    for train, (ok, reason) in availability.items():
        if not ok:
            print(f"[pulando treino {train}] {reason}")
    if skipped:
        print(f"{len(skipped)} jobs já concluídos (resume)")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("--families", type=str, default="all",
                        help="all ou lista: ae,skip,vae,joint,jointskip")
    parser.add_argument("--bases", type=str, default=",".join(BASES))
    parser.add_argument("--indices", type=str, default="all", help="all | 0-9 | 0,3,7")
    parser.add_argument("--trains", type=str, default="all",
                        help="all ou lista de datasets de treino (ex.: PUC,camera1)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--mlflow", action="store_true",
                        help="Loga no servidor MLflow (Config.IP_LOCAL); default: ./mlruns local")
    args = parser.parse_args()

    jobs = build_jobs(args)
    print(f"Total de jobs de classificador (5 batches cada): {len(jobs)}")
    if not jobs:
        return

    devices = unique_devices()
    n_procs = min(len(devices), len(jobs))

    if n_procs <= 1 or devices[0] == "cpu":
        worker(0, [jobs], devices, args.epochs, args.mlflow, not args.no_resume)
        return

    jobs_split = split_jobs(jobs, n_procs)
    print(f"Processos: {n_procs} | Jobs por processo: {[len(j) for j in jobs_split]}")
    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker,
        args=(jobs_split, devices, args.epochs, args.mlflow, not args.no_resume),
        nprocs=n_procs, join=True,
    )


if __name__ == "__main__":
    main()
