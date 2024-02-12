import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from models.lstm import LSTM_net


class CancerSequences(Dataset):
    def __init__(self, csv_file_path, inputs, outputs):
        super().__init__()

        self.sequence_length = 50
        self.data_path = os.path.join(os.getcwd(), "bsc_pathways/pathway_enrichment_v1/data/saved_tensors")
        inputs_filepath = os.path.join(self.data_path, f"{inputs}.pt")
        outputs_filepath = os.path.join(self.data_path, f"{outputs}.pt")

        if os.path.isfile(inputs_filepath) and os.path.isfile(outputs_filepath):
            self.inputs = torch.load(inputs_filepath)
            self.outputs = torch.load(outputs_filepath)
        else:
            le = LabelEncoder()
            df = pd.read_csv(csv_file_path)
            enriched_pathways = torch.tensor(df.iloc[:, 4:33].values).float()
            gene_expressions = torch.tensor(df.iloc[:, 33:8987].values).float()
            stage_transitions = torch.tensor(
                le.fit_transform(df.iloc[:, 1].values.astype(str))
            ).float()

            torch.save(
                gene_expressions, os.path.join(self.data_path, "gene_expressions.pt")
            )
            torch.save(
                enriched_pathways, os.path.join(self.data_path, "enriched_pathways.pt")
            )
            torch.save(
                stage_transitions, os.path.join(self.data_path, "stage_transitions.pt")
            )

            self.inputs = locals()[inputs]
            self.outputs = locals()[outputs]

    def __len__(self):
        return int(len(self.outputs) / self.sequence_length)

    def __getitem__(self, idx):
        start_row = idx * self.sequence_length
        end_row = (idx + 1) * self.sequence_length

        return (
            self.inputs[start_row:end_row],
            self.outputs[start_row],
        )


csv_file_path = os.path.join(os.getcwd(), f"bsc_pathways/pathway_enrichment_v1/data/lobular_w_negatives.csv")
dataset = CancerSequences(
    csv_file_path, inputs="enriched_pathways", outputs="stage_transitions"
)


train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)

num_epochs = 300
lstm = LSTM_net(input_size=29, hidden_size=512, num_layers=1)
optimizer = optim.Adam(lstm.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

metrics = {
    "f1-macro": MulticlassF1Score(
        num_classes=5,
        average="macro",
    ),
    "f1-micro": MulticlassF1Score(
        num_classes=5,
        average="micro",
    ),
    "accuracy": MulticlassAccuracy(
        num_classes=5,
    ),
}


train_losses = []
test_losses = []
for epoch in range(num_epochs):
    train_loss = lstm.run_dataloader(
        train_dl, epoch, num_epochs, optimizer, loss_function, metrics, train=True
    )
    test_loss = lstm.run_dataloader(
        test_dl, epoch, num_epochs, optimizer, loss_function, metrics, train=False
    )

    train_losses.append(train_loss)
    test_losses.append(test_loss)