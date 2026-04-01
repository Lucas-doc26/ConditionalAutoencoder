import os
import csv
import numpy as np
import pandas as pd
from scipy.special import softmax
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score
from itertools import combinations


def fusion_sum(stacked):
    agg = np.sum(stacked, axis=0)
    return agg / agg.sum(axis=1, keepdims=True)


def fusion_mean(stacked):
    agg = np.mean(stacked, axis=0)
    return agg / agg.sum(axis=1, keepdims=True)


def fusion_max(stacked):
    agg = np.max(stacked, axis=0)
    return agg / agg.sum(axis=1, keepdims=True)


def fusion_mult(stacked):
    logp = np.log(np.clip(stacked, 1e-300, 1.0))
    sumlog = np.sum(logp, axis=0)
    exp = np.exp(sumlog - np.max(sumlog, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def fusion_vote(stacked):
    n_models, N, C = stacked.shape
    preds_models = np.argmax(stacked, axis=2)
    final_preds = np.zeros(N, dtype=np.int64)

    for i in range(N):
        votes = preds_models[:, i]
        counts = Counter(votes)
        top = counts.most_common()
        max_votes = top[0][1]
        tied = [cls for cls, cnt in top if cnt == max_votes]

        if len(tied) == 1:
            final_preds[i] = tied[0]
        else:
            probs_sum = np.sum(stacked[:, i, :], axis=0)
            final_preds[i] = max(tied, key=lambda c: probs_sum[c])

    probs = np.eye(C)[final_preds]
    return probs.astype(np.float32)


nums = list(range(10))


def stack_predictions(models_outputs):
    # models_outputs: lista de arrays [N, C]

    stacked = np.stack(models_outputs, axis=0)  # [M, N, C]

    return stacked


def check_all_models_npy_same_length(path_npy_template, n_models=10):
    """
    Verifica se todos os arquivos batches-*.npy dos modelos 0..n_models-1
    existem e têm o mesmo número de linhas (amostras).
    Retorna (ok, info): info é {model_id: n_samples} ou {"error": "..."}.
    """
    lengths = {}
    for i in range(n_models):
        p = path_npy_template.replace("Z", str(i))
        if not os.path.isfile(p):
            return False, {"error": f"arquivo ausente: {p}"}
        arr = np.load(p, mmap_mode="r")
        lengths[i] = int(arr.shape[0])
    if len(set(lengths.values())) != 1:
        return False, lengths
    return True, lengths


def make_fusion(n_models, path_npy, y_true):

    results = {}
    
    combinacoes = list(combinations(nums, n_models))
    for comb in combinacoes:
        stacked = []
        for n_model in comb:
            npy = np.load(path_npy.replace('Z', str(n_model)))
            stacked.append(npy)
        stacked_preds = stack_predictions(stacked)
        
        accuracy_score_dict = {}
        for name_func, func in fusion_methods.items():
            y_pred = func(stacked_preds)
            y_pred = np.argmax(y_pred, axis=1)
            accuracy_score_dict[name_func] = accuracy_score(y_true, y_pred)
        results[comb] = accuracy_score_dict    
    
    return results

fusion_methods = {
    "sum": fusion_sum,
    "mean_probs": fusion_mean,
    "max": fusion_max,
    "mult": fusion_mult,
    "vote": fusion_vote
}

dict_combinations = {
    #"PKLot": ['camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9'],
    "CNR": ['UFPR04', 'UFPR05', 'PUC'],
    'Kyoto': ['UFPR04', 'UFPR05', 'PUC', 'camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']
}

testes = ['UFPR04', 'UFPR05', 'PUC', 'camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6', 'camera7', 'camera8', 'camera9']
batches = [64, 128, 256, 512, 1024]


rows = []

for encoder, classificadores in dict_combinations.items():
    print(encoder)
    for classificador in classificadores:
        for teste in testes:                    
            for batch in batches:
                path_numpy_dir = f'/home/lucas/representation-fusion/Modelos/Modelo_Kyoto-0/Classificador-{encoder}/Resultados/Treinados_em_{classificador}/{teste}'
                if os.path.isdir(path_numpy_dir):

                    path_npy = f'/home/lucas/representation-fusion/Modelos/Modelo_Kyoto-Z/Classificador-{encoder}/Resultados/Treinados_em_{classificador}/{teste}/batches-{batch}.npy'
                    ok_lens, lens = check_all_models_npy_same_length(path_npy)
                    if not ok_lens:
                        print(
                            f"[skip] tamanhos .npy inconsistentes ou faltando — "
                            f"encoder={encoder} classificador={classificador} teste={teste} batch={batch} | {lens}"
                        )
                        continue

                    y_true1 = pd.read_csv(f'/home/lucas/representation-fusion/CSV_antigo/{teste}/{teste}_test.csv')['class']
                    y_true2 = pd.read_csv(f'/home/lucas/representation-fusion/CSV_antigo/{teste}/{teste}.csv')['class']

                    n_samples = int(next(iter(lens.values())))
                    if y_true1.shape[0] == n_samples:
                        y_true = y_true1
                    elif y_true2.shape[0] == n_samples:
                        y_true = y_true2
                    else:
                        print(
                            f"[skip] nenhum CSV de rótulo com {n_samples} linhas — "
                            f"encoder={encoder} classificador={classificador} teste={teste} batch={batch} | "
                            f"test={y_true1.shape[0]} full={y_true2.shape[0]}"
                        )
                        continue

                    for n in range(2, 11, 1):
                        print(n)
                        fusions = make_fusion(n, path_npy, y_true)
                        for comb, fusion in fusions.items():
                            for f_name, value in fusion.items():
                                rows.append({
                                    'encoder': encoder,
                                    'classificador': classificador,
                                    'teste': teste,
                                    'batch': batch,
                                    'n_models': n,
                                    'combination': comb,
                                    'fusion_method': f_name,
                                    'value': value
                                })       
                                
data = pd.DataFrame(rows)
data.to_csv("/home/lucas/representation-fusion/combinacoes.csv", index=False)           