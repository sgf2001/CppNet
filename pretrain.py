import pandas as pd
import numpy as np
import torch.nn as nn
import esm
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='esm2_t33_650M_UR50D')
parser.add_argument('--layer', type=int, default=33)
args = parser.parse_args()

model_path = f'esm.pretrained.{args.model}()'
model, alphabet = eval(model_path)
batch_converter = alphabet.get_batch_converter()
model.eval()

def genpepData(file_path):
    data = pd.read_csv(file_path)
    peptide = data.iloc[:, 0]
    return peptide

peptide = genpepData('./data/dataset.csv').tolist()

chunk_size = 1
my_data =[]

for i in range(0, len(peptide), chunk_size):
    sequences = peptide[i:i + chunk_size]
    data_list = []
    for j in range(0,len(sequences)):
        data_list.append((1, sequences[j][:len(sequences[j])]))

    batch_labels, batch_strs, batch_tokens = batch_converter(data_list)#padding
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)#real lens

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[args.layer], return_contacts=True)
    token_representations = results["representations"][args.layer]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    my_data.append(sequence_representations)

peptide_all_data =[]
for i in my_data:
    for j in i:
        j =j.tolist()
        peptide_all_data.append(j)
peptide_data = np.array(peptide_all_data)
peptide_feature = pd.DataFrame(peptide_data)
peptide_feature.to_csv('dataset_esm_2_feature.csv')
