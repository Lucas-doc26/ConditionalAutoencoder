import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────── constants ────────────────────────────────────────

BASE_DIR  = "/home/lucas.ocunha/Downloads/PIBIC-ANO-PASSADO/Modelos"
OUT_DIR   = "/home/lucas.ocunha/ConditionalAutoencoder/resultados_pibic_ano_passado"
OUT_CSV   = os.path.join(OUT_DIR, "precisao.csv")

N_MODELS  = 10
BATCHES   = [64, 128, 256, 512, 1024]

ENCODERS = ["PKLot", "CNR", "Kyoto"]

CLASSIFICADORES = {
    "PKLot": ["camera1", "camera2", "camera3", "camera4", "camera5",
              "camera6", "camera7", "camera8", "camera9"],
    "CNR":   ["UFPR04", "UFPR05", "PUC"],
    "Kyoto": ["UFPR04", "UFPR05", "PUC",
              "camera1", "camera2", "camera3", "camera4", "camera5",
              "camera6", "camera7", "camera8", "camera9"],
}

TESTES = ["UFPR04", "UFPR05", "PUC",
          "camera1", "camera2", "camera3", "camera4", "camera5",
          "camera6", "camera7", "camera8", "camera9"]

# ─────────────────────────── worker ───────────────────────────────────────────

def process_file(model_id, encoder, classificador, teste):
    """
    Reads one precisao-{teste}.txt and returns a list of row-dicts.
    Returns [] if the file doesn't exist or has unexpected format.
    """
    path = (
        f"{BASE_DIR}/Modelo_Kyoto-{model_id}/"
        f"Classificador-{encoder}/Precisao/"
        f"Treinado_em_{classificador}/precisao-{teste}.txt"
    )

    if not os.path.isfile(path):
        return []

    try:
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) != len(BATCHES):
            logger.warning(f"[skip] expected {len(BATCHES)} lines, got {len(lines)}: {path}")
            return []

        rows = []
        for batch, value in zip(BATCHES, lines):
            rows.append({
                "modelo":        f"Modelo{model_id + 1}",
                "encoder":       encoder,
                "classificador": classificador,
                "teste":         teste,
                "batch":         batch,
                "value":         float(value),
            })
        return rows

    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return []


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build all (model_id, encoder, classificador, teste) combos
    tasks = [
        (model_id, encoder, classificador, teste)
        for model_id in range(N_MODELS)
        for encoder, classificadores in CLASSIFICADORES.items()
        for classificador in classificadores
        for teste in TESTES
    ]
    logger.info(f"Total file tasks: {len(tasks)}")

    all_rows = []

    # I/O-bound → ThreadPoolExecutor is ideal here
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_file, *t): t for t in tasks}

        for i, fut in enumerate(as_completed(futures), 1):
            task = futures[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
            except Exception as e:
                logger.error(f"Task {task} raised: {e}")

            if i % 200 == 0 or i == len(tasks):
                logger.info(f"[{i}/{len(tasks)}] rows so far: {len(all_rows)}")

    df = pd.DataFrame(all_rows, columns=["modelo", "encoder", "classificador", "teste", "batch", "value"])
    df = df.sort_values(["modelo", "encoder", "classificador", "teste", "batch"]).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    logger.info(f"Done. Saved {len(df)} rows → {OUT_CSV}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()