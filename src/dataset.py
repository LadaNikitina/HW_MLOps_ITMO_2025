from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class NTDataset(Dataset):
    def __init__(self, sequences, labels, k=6):
        self.sequences = sequences
        self.labels = labels
        self.k = k

    def __len__(self):
        return len(self.sequences)

    def _to_kmers(self, seq):
        return " ".join([seq[i : i + self.k] for i in range(len(seq) - self.k + 1)])

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return sequence, self.labels[idx]


def load_nucleotide_transformer(
    batch_size, valid_split=-1, dataset_name="enhancers", split_state=42
):
    train_dataset = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        dataset_name,
        split="train",
        streaming=False,
    )

    test_dataset = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        dataset_name,
        split="test",
        streaming=False,
    )

    train_sequences = train_dataset["sequence"]
    train_labels = train_dataset["label"]

    if valid_split > 0:
        train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(
            train_sequences, train_labels, test_size=valid_split, random_state=split_state
        )

    test_sequences = test_dataset["sequence"]
    test_labels = test_dataset["label"]

    train_dataset = NTDataset(train_sequences, train_labels)
    test_dataset = NTDataset(test_sequences, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    if valid_split > 0:
        validation_dataset = NTDataset(validation_sequences, validation_labels)
        valid_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        print(
            f"Train: {len(train_loader.dataset)}, Validation: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}"
        )
        return train_loader, valid_loader, test_loader

    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
    return train_loader, None, test_loader
