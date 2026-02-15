import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import os 
import mlflow 
import mlflow.pytorch
import torch.multiprocessing as mp
from tqdm import tqdm

from src.config import *
from src.utils.datasets import CustomImageDataset
from src.utils.plot import plot_reconstruction, denormalize
from src.utils.transform import return_transform
from src.models import *
from src.utils.image_metrics import calculate_all_metrics_torch

def train_experiment_autoencoder(
    gpu_id=0, 
    model_class=None, 
    dataset_name=None,
    batch_size=32,
    num_epochs=10,
    lr=1e-3
):
    device = Config.DEVICES[gpu_id]
    torch.cuda.set_device(device)

    
    transform = return_transform()  
    
    train_dataset = CustomImageDataset(
        f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{dataset_name}/{dataset_name}_autoencoder_train.csv",
        transform=transform
    )
    val_dataset = CustomImageDataset(
        f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{dataset_name}/{dataset_name}_autoencoder_validation.csv",
        transform=transform
    )
    test_dataset = CustomImageDataset(
        f"/home/lucas.ocunha/ConditionalAutoencoder/CSV/{dataset_name}/{dataset_name}_autoencoder_test.csv",
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False)
    
    model_name = model_class.__name__
    mlflow.set_experiment(model_name)

    with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):

        # -------------------------
        # Params
        # -------------------------
        mlflow.log_param("model", model_name)
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("lr", lr)
        mlflow.log_param("loss", "MSE")
        mlflow.log_param("input_shape", "3x128x128")

        model = model_class().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # -------------------------
        # Train / Val
        # -------------------------
        for epoch in range(num_epochs):

            pbar = tqdm(
                train_loader,
                desc=f"Epoch [{epoch+1}/{num_epochs}]",
                leave=False
            )

            model.train()
            train_loss = 0.0
            for x, y in pbar:
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = criterion(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    val_loss += criterion(model(x), y).item()

            val_loss /= len(val_loader)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # -------------------------
        # Test + Plot
        # -------------------------
        model.eval()
        with torch.no_grad():
            for x, _ in test_loader:
                x = x[:8].to(device)
                recon = model(x)
                break

        plot_path = plot_reconstruction(
            x, recon, model_name, dataset_name
        )
        mlflow.log_artifact(plot_path, artifact_path="reconstructions")


        metrics_sum = {
            "MSE": 0.0,
            "SSIM": 0.0,
            "PSNR": 0.0,
            "NCC": 0.0,
            "VIF": 0.0,
            "SCC": 0.0
        }

        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):

                if i >= 2:
                    break

                images = images.to(device)
                outputs = model(images)

                images_dn = torch.clamp(denormalize(images), 0, 1)
                outputs_dn = torch.clamp(denormalize(outputs), 0, 1)

                batch_metrics = calculate_all_metrics_torch(
                    images_dn, outputs_dn
                )

                for k in metrics_sum:
                    metrics_sum[k] += batch_metrics[k]
        

        metrics_avg = {
            k: float(np.mean(v)) for k, v in metrics_sum.items()
        }
        for name, value in metrics_avg.items():
            mlflow.log_metric(f"test_{name.lower()}", value)


        # -------------------------
        # Models
        # -------------------------
        mlflow.pytorch.log_model(model, "autoencoder")

        if hasattr(model, "encoder"):
            mlflow.pytorch.log_model(model.encoder, "encoder")
        if hasattr(model, "decoder"):
            mlflow.pytorch.log_model(model.decoder, "decoder")

        print(f"✓ {model_name} | {dataset_name} finalizado")

        os.makedirs(f"models/{model_name}_{dataset_name}", exist_ok=True)

        torch.save(model.state_dict(), f"models/{model_name}_{dataset_name}/autoencoder.pth")
        torch.save(model.encoder.state_dict(), f"models/{model_name}_{dataset_name}/encoder.pth")
        torch.save(model.decoder.state_dict(), f"models/{model_name}_{dataset_name}/decoder.pth")


def worker(rank, jobs_split):
    NUM_GPUS = len(Config.DEVICES)
    gpu_id = rank % NUM_GPUS

    torch.cuda.set_device(gpu_id)

    mlflow.set_tracking_uri(Config.IP_LOCAL)

    my_jobs = jobs_split[rank]

    print(f"[Rank {rank}] -> GPU {gpu_id} | {len(my_jobs)} jobs")

    for model, dataset_encoder, epochs in my_jobs:
        train_experiment_autoencoder(
            model_class=model,
            dataset_name=dataset_encoder,
            num_epochs=epochs,
            gpu_id=int(gpu_id)
        )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Número de épocas para treinamento")
    parser.add_argument("-v", "--models", type=str, default="all", help="Usar todos os autoencoders")    

    
    args = parser.parse_args()
    
    epochs = args.epochs

    if args.models == "all":
        encoders = [
            Autoencoder0, Autoencoder1, Autoencoder2,
            Autoencoder3, Autoencoder4, Autoencoder5,
            Autoencoder6, Autoencoder7, Autoencoder8,
            Autoencoder9,
            SkipAutoencoder0, SkipAutoencoder1, SkipAutoencoder2,
            SkipAutoencoder3, SkipAutoencoder4, SkipAutoencoder5,
            SkipAutoencoder6, SkipAutoencoder7, SkipAutoencoder8,
            SkipAutoencoder9,
            VariationalAutoencoder0, VariationalAutoencoder1, VariationalAutoencoder2,
            VariationalAutoencoder3, VariationalAutoencoder4, VariationalAutoencoder5,
            VariationalAutoencoder6, VariationalAutoencoder7, VariationalAutoencoder8,
            VariationalAutoencoder9
        ]

    elif args.models == "variational":
        encoders = [
            VariationalAutoencoder0, VariationalAutoencoder1, VariationalAutoencoder2,
            VariationalAutoencoder3, VariationalAutoencoder4, VariationalAutoencoder5,
            VariationalAutoencoder6, VariationalAutoencoder7, VariationalAutoencoder8,
            VariationalAutoencoder9
        ]
    elif args.models == "skip":
        encoders = [
            SkipAutoencoder0, SkipAutoencoder1, SkipAutoencoder2,
            SkipAutoencoder3, SkipAutoencoder4, SkipAutoencoder5,
            SkipAutoencoder6, SkipAutoencoder7, SkipAutoencoder8,
            SkipAutoencoder9
        ]
    else:
        encoders = [model for model in globals().values() if isinstance(model, type) and issubclass(model, Autoencoder) and model != Autoencoder]
        raise ValueError(f"Modelo inválido: {args.models}")

    jobs = []
    n_procs = len(Config.DEVICES)

    for dataset_encoder in ["CNR", "PKLot"]:
        for model in encoders:
            jobs.append((model, dataset_encoder, epochs))
            
    jobs.sort(key=lambda x: x[0].__name__)

    NUM_GPUS = len(Config.DEVICES)     # ex: 2
    PROCS_PER_GPU = 5                 # ~3GB por processo

    n_procs = NUM_GPUS * PROCS_PER_GPU
    n_procs = min(n_procs, len(jobs))

    jobs_split = split_jobs(jobs, n_procs)

    print(f"Total de jobs: {len(jobs)}")
    print(f"Processos: {n_procs}")
    print(f"Jobs por processo: {[len(j) for j in jobs_split]}")

    mp.set_start_method("spawn", force=True)
    mp.spawn(
        worker,
        args=(jobs_split,),
        nprocs=n_procs,
        join=True
    )