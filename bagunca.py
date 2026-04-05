import os
import pandas as pd 

base_path = '/home/lucas.ocunha/Downloads/PIBIC-ANO-PASSADO/CSV'

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith('.csv'):
            full_path = os.path.join(root, file)

            df = pd.read_csv(full_path)

            df['path_image'] = df['path_image'].str.replace(
                '/datasets/PKLot',
                '/home/lucas.ocunha/ConditionalAutoencoder/PKLot',
                regex=False
            )

            df['path_image'] = df['path_image'].str.replace(
                '/datasets/CNR',
                '/home/lucas.ocunha/DeepLearning/datasets/CNR',
                regex=False
            )

            df.to_csv(full_path, index=False)

            print(f'Atualizado: {full_path}')