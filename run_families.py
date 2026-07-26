"""Orquestrador do pipeline por famílias: csv-fix -> ae -> cls -> fusion.

Cada estágio é idempotente (retoma de onde parou); pré-checagens abortam com
mensagem acionável quando faltam artefatos do estágio anterior.

Uso:
    python run_families.py --stage all -e 10 --families all
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PY = sys.executable

STAGES = ["csv-fix", "ae", "cls", "fusion"]


def run(cmd):
    print(f"\n=== {' '.join(cmd)} ===")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def stage_csv_fix(args):
    run([PY, "scripts/make_kyoto_dataaug.py"])
    run([PY, "scripts/fix_csv_paths.py"])
    run([PY, "scripts/verify_test_alignment.py", "--canonize"])


def stage_ae(args):
    cmd = [PY, "-m", "src.pipeline.train_autoencoders_families",
           "-e", str(args.epochs), "--families", args.families,
           "--bases", args.bases, "--indices", args.indices]
    if args.no_resume:
        cmd.append("--no-resume")
    if args.mlflow:
        cmd.append("--mlflow")
    run(cmd)


def stage_cls(args):
    if not list(PROJECT_ROOT.glob("Modelos/Modelo_*/Modelo-Base/Pesos/*_encoder.pth")):
        sys.exit("Nenhum encoder .pth em Modelos/*/Modelo-Base/Pesos — rode antes: "
                 "python run_families.py --stage ae")
    cmd = [PY, "-m", "src.pipeline.train_classifiers_families",
           "-e", str(args.epochs), "--families", args.families,
           "--bases", args.bases, "--indices", args.indices]
    if args.no_resume:
        cmd.append("--no-resume")
    if args.mlflow:
        cmd.append("--mlflow")
    run(cmd)


def stage_fusion(args):
    npys = list(PROJECT_ROOT.glob("Modelos/Modelo_*/Classificador-*/Resultados/**/batches-*.npy"))
    if not npys:
        sys.exit("Nenhum .npy em Modelos/*/Classificador-*/Resultados — rode antes: "
                 "python run_families.py --stage cls")
    # No estágio de fusão, as famílias são os prefixos das pastas (Modelo_AE,...).
    fusion_families = args.fusion_families or "all"
    run([PY, "fusion_families.py", "--families", fusion_families,
         "--frac", str(args.frac)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=str, default="all",
                        help="all | " + " | ".join(STAGES))
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("--families", type=str, default="all",
                        help="all ou lista: ae,skip,vae,joint,jointskip")
    parser.add_argument("--fusion-families", type=str, default="",
                        help="Famílias no estágio de fusão (prefixos: Modelo_AE,...); default all")
    parser.add_argument("--bases", type=str, default="CNR,PKLot,Kyoto")
    parser.add_argument("--indices", type=str, default="all")
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    stages = STAGES if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage not in STAGES:
            sys.exit(f"Estágio inválido: {stage}. Use all | {' | '.join(STAGES)}")

    handlers = {
        "csv-fix": stage_csv_fix,
        "ae": stage_ae,
        "cls": stage_cls,
        "fusion": stage_fusion,
    }
    for stage in stages:
        handlers[stage](args)


if __name__ == "__main__":
    main()
