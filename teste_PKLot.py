import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# ───────────────────────── CONFIGURE AQUI ─────────────────────────

encoder       = "PKLot"
classificador = "camera1"
teste         = "camera1"   # ← TESTE PROBLEMÁTICO
batch         = 128
model_id      = 0          # testa 1 modelo só

BASE_MODEL_DIR = "/home/lucas.ocunha/Downloads/PIBIC-ANO-PASSADO/Modelos"
BASE_CSV_DIR   = "/home/lucas.ocunha/Downloads/PIBIC-ANO-PASSADO/CSV"

# ───────────────────────── PATHS ─────────────────────────

def npy_path():
    return (
        f"{BASE_MODEL_DIR}/Modelo_Kyoto-{model_id}/"
        f"Classificador-{encoder}/Resultados/"
        f"Treinados_em_{classificador}/{teste}/batches-{batch}.npy"
    )
#/home/lucas.ocunha/Downloads/PIBIC-ANO-PASSADO/Modelos/Modelo_Kyoto-0/Classificador-PKLot/Resultados/Treinados_em_camera1/camera1/batches-64.npy
def csv_path():
    for suffix in ["_test.csv", ".csv"]:
        p = f"{BASE_CSV_DIR}/{teste}/{teste}{suffix}"
        if os.path.isfile(p):
            return p
    return None

# ───────────────────────── LOAD ─────────────────────────

npy_file = npy_path()
csv_file = csv_path()

print("\n===== PATHS =====")
print("NPY:", npy_file)
print("CSV:", csv_file)

if not os.path.isfile(npy_file):
    raise FileNotFoundError("NPY não encontrado")

if csv_file is None:
    raise FileNotFoundError("CSV não encontrado")

arr = np.load(npy_file)
y_true = pd.read_csv(csv_file)["class"].values

print("\n===== SHAPES =====")
print("arr:", arr.shape)
print("y_true:", y_true.shape)

# ───────────────────────── TESTE 1: MODELO INDIVIDUAL ─────────────────────────

preds = np.argmax(arr, axis=1)

acc = accuracy_score(y_true, preds)
print("\n===== ACCURACY =====")
print("Normal:", acc)

# ───────────────────────── TESTE 2: FLIP ─────────────────────────

acc_flip = accuracy_score(y_true, 1 - preds)
print("Flip:", acc_flip)

# ───────────────────────── TESTE 3: INSPEÇÃO ─────────────────────────

print("\n===== SAMPLE =====")
print("y_true[:20]:", y_true[:20])
print("preds [:20]:", preds[:20])

# ───────────────────────── TESTE 4: PROBABILIDADES ─────────────────────────

print("\n===== PROBABILIDADES (primeiras 5) =====")
print(arr[:5])

# ───────────────────────── DIAGNÓSTICO AUTOMÁTICO ─────────────────────────

print("\n===== DIAGNÓSTICO =====")

if not teste.startswith("camera") and encoder == "PKLot":
    print("❌ PROVÁVEL ERRO: PKLot sendo testado em outro dataset (UFPR/PUC)")

if acc < 0.6 and acc_flip > acc:
    print("⚠️ POSSÍVEL: classes invertidas")

if acc < 0.6 and acc_flip < 0.6:
    print("⚠️ POSSÍVEL: desalinhamento entre NPY e CSV")

if acc > 0.8:
    print("✅ Modelo OK → problema está no ensemble")