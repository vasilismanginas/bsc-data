import os
import torch
import pandas as pd
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class GenesToPathways(Dataset):
    def __init__(self):
        df = pd.read_csv(
            os.path.join(
                os.getcwd(),
                f"bsc_pathways/pathway_enrichment_v1/data/lobular_w_negatives.csv",
            )
        )

        self.gene_expressions = torch.tensor(df.iloc[:, 33:8987].values).float()
        self.enriched_pathways = torch.tensor(df.iloc[:, 4:33].values).float()
        print()

    def __len__(self):
        return len(self.gene_expressions)

    def __getitem__(self, idx):
        return (self.gene_expressions[idx], self.enriched_pathways[idx])


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=8954, out_features=100),
            nn.Linear(in_features=100, out_features=100),
            nn.Linear(in_features=100, out_features=29),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)

    def run_dataloader(
        self, dataloader, epoch, num_epochs, optimizer, loss_function, metrics, train
    ):
        progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("loss: {task.fields[loss]:.4f}"),
            *[
                TextColumn(
                    "{}: {{task.fields[{}]:.2f}}".format(metric_name, metric_name)
                )
                for metric_name in metrics
            ],
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("(test)" if not train else ""),
        )
        epoch_task = progress_bar.add_task(
            "epoch: [{}/{}]".format(epoch, num_epochs), total=len(dataloader)
        )
        progress_bar.update(
            task_id=epoch_task,
            advance=0,
            **{metric: 0 for metric in metrics}, # type: ignore
            loss=0,
        )

        with progress_bar as progress:
            for sequence_id, (inputs, labels) in enumerate(dataloader):
                outputs = self(inputs)

                loss = loss_function(outputs, labels)
                # loss[labels == 1.0] *= 8
                # loss = loss.mean()

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    sequence_metrics = {
                        metric_name: (
                            (
                                sequence_id
                                * progress.tasks[epoch_task].fields[metric_name]
                            )
                            + metric(outputs, labels).item()
                        )
                        / (sequence_id + 1)
                        for metric_name, metric in metrics.items()
                    }

                progress.update(
                    task_id=epoch_task,
                    advance=1,
                    **sequence_metrics,
                    loss=(
                        (sequence_id * progress.tasks[epoch_task].fields["loss"])
                        + loss.item()
                    )
                    / (sequence_id + 1),
                )
        return progress.tasks[epoch_task].fields["loss"]


dataset = GenesToPathways()

train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
train_dl = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=True)

mlp_net = MLP()
num_epochs = 50
optimizer = optim.Adam(mlp_net.parameters(), lr=1e-4)
loss_function = nn.BCELoss()

metrics = {
    "f1-macro": torchmetrics.F1Score(
        task="multilabel",
        average="macro",
        num_labels=29,
    ),
    "f1-micro": torchmetrics.F1Score(
        task="multilabel",
        average="micro",
        num_labels=29,
    ),
    "accuracy": torchmetrics.Accuracy(
        task="multilabel",
        num_labels=29,
    ),
}


train_losses = []
test_losses = []
for epoch in range(num_epochs):
    train_loss = mlp_net.run_dataloader(
        train_dl, epoch, num_epochs, optimizer, loss_function, metrics, train=True
    )
    test_loss = mlp_net.run_dataloader(
        test_dl, epoch, num_epochs, optimizer, loss_function, metrics, train=False
    )

    train_losses.append(train_loss)
    test_losses.append(test_loss)