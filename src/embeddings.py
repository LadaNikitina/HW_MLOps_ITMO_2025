import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dnabert2_embeddings(dataloader, tokenizer, model, pooling="mean"):
    embeddings = []
    labels = []
    model.to(device)

    for batch in tqdm(dataloader):
        sequences, lbls = batch

        tokens = tokenizer(
            list(sequences),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)
            hidden_states = outputs[0]

        if pooling == "mean":
            pooled = hidden_states.mean(dim=1)
        elif pooling == "max":
            pooled = hidden_states.max(dim=1)[0]
        else:
            raise ValueError("pooling must be 'mean' or 'max'")

        embeddings.append(pooled.cpu().numpy())
        labels.append(lbls.numpy())

    return np.vstack(embeddings), np.concatenate(labels)