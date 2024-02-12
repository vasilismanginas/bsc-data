import torch
import torch.nn as nn
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class LSTM_net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, (hidden_state, _) = self.lstm(x)
        output = self.mlp(hidden_state)
        return output.flatten()

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
            **{metric: 0 for metric in metrics},
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


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
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
            TextColumn("loss: {task.fields[loss]:.2f}"),
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
            **{metric: 0 for metric in metrics},
            loss=0,
        )

        with progress_bar as progress:
            for sequence_id, (inputs, labels) in enumerate(dataloader):
                outputs = self(inputs)
                loss = loss_function(outputs, labels)

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
