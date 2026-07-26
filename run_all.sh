#!/usr/bin/env bash
# Ponto de entrada único do pipeline por famílias de autoencoder.
#
# Configurável por variáveis de ambiente:
#   DATASETS_ROOT  raiz dos datasets (default: $HOME/DeepLearning/datasets)
#                  (também: CNR_DIR, PKLOT_DIR, KYOTO_DIR para paths individuais)
#   EPOCHS         épocas de treino (default: 10)
#   FAMILIES       famílias a treinar (default: all = ae,skip,vae,joint,jointskip)
#   FRAC           fração das combinações na fusão (default: 1.0 = exaustivo)
#   PYTHON         interpretador (default: python3)
#
# Todos os estágios são idempotentes: pode re-executar este script após uma
# interrupção (ou depois de extrair o PKLot) que ele só processa o que falta.

set -euo pipefail
cd "$(dirname "$0")"

export DATASETS_ROOT="${DATASETS_ROOT:-$HOME/DeepLearning/datasets}"
EPOCHS="${EPOCHS:-10}"
FAMILIES="${FAMILIES:-all}"
FRAC="${FRAC:-1.0}"
PYTHON="${PYTHON:-python3}"

echo "DATASETS_ROOT=$DATASETS_ROOT | EPOCHS=$EPOCHS | FAMILIES=$FAMILIES | FRAC=$FRAC"

# 1. Dados: dataAug do Kyoto, correção de paths nos CSVs, canonização dos testes
"$PYTHON" run_families.py --stage csv-fix

# 2. Pré-treino dos autoencoders (famílias × índices × bases)
"$PYTHON" run_families.py --stage ae -e "$EPOCHS" --families "$FAMILIES"

# 3. Classificadores + geração dos .npy de probabilidade
"$PYTHON" run_families.py --stage cls -e "$EPOCHS" --families "$FAMILIES"

# 4. Fusão: varredura de combinações 2..10 dentro de cada família
"$PYTHON" run_families.py --stage fusion --frac "$FRAC"

# 5. Fusão cross-família (1 modelo por família)
"$PYTHON" fusion_families.py --cross --frac "$FRAC"

echo "Pipeline concluído."
