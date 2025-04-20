import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_nt_embeddings(dataloader, tokenizer, model, pooling="mean"):
    embeddings = []
    labels = []

    model.to(device)
    model.eval()

    for batch in tqdm(dataloader, desc="Generating embeddings"):
        sequences, lbls = batch

        # Токенизация
        tokens = tokenizer(
            list(sequences), padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Инференс с отключением градиентов и автокастом
        with torch.no_grad():
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
                )

            # Используем последний скрытый слой
            last_hidden = outputs.hidden_states[-1]  # (B, T, H)

            if pooling == "mean":
                attention_mask_exp = attention_mask.unsqueeze(-1)  # (B, T, 1)
                sum_embeddings = torch.sum(last_hidden * attention_mask_exp, dim=1)  # (B, H)
                sum_mask = attention_mask_exp.sum(dim=1).clamp(min=1e-9)  # (B, 1)
                pooled = sum_embeddings / sum_mask
            elif pooling == "max":
                pooled = last_hidden.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
                pooled = torch.max(pooled, dim=1).values  # (B, H)
            else:
                raise ValueError("pooling must be 'mean' or 'max'")

        embeddings.append(pooled.cpu().numpy())
        labels.append(lbls.numpy())

    return np.vstack(embeddings), np.concatenate(labels)
