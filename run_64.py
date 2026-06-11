import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Pipeline 64x64 completo")
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Número de épocas para treinamento")
    parser.add_argument("-m", "--models", type=str, default="all",
                        help="all | vae | skip | ae | joint (apenas para o treino de autoencoders)")
    parser.add_argument("--skip-ae",     action="store_true",
                        help="Pula treino de autoencoders")
    parser.add_argument("--skip-cls",    action="store_true",
                        help="Pula treino de classifiers")
    parser.add_argument("--skip-fusion", action="store_true",
                        help="Pula fusão dos resultados")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_ae:
        print("=== [1/3] Treinando autoencoders 64x64 ===")
        subprocess.run(
            [py, "src/pipeline/train_autoencoders_64.py",
             "-e", str(args.epochs), "-m", args.models],
            check=True
        )

    if not args.skip_cls:
        print("=== [2/3] Treinando classifiers 64x64 ===")
        subprocess.run(
            [py, "src/pipeline/train_classifier_64.py",
             "-e", str(args.epochs)],
            check=True
        )

    if not args.skip_fusion:
        print("=== [3/3] Fusão dos resultados 64x64 ===")
        subprocess.run(
            [py, "src/pipeline/fusion_64.py"],
            check=True
        )


if __name__ == "__main__":
    main()
