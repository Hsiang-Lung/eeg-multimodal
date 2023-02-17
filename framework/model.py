import torch.nn as nn
import torch
from tqdm import tqdm
from torchinfo import summary
from framework.utils import plot_roc


# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9696316
class Model(nn.Module):
    def __init__(self, num_classes=2, sequence_len=5500, with_table=False):
        super().__init__()
        self.with_table = with_table

        def calcOut(input, kernel, padding=0, stride=1, dialation=1):
            return int((input + 2 * padding - dialation * (kernel - 1) - 1) / stride + 1)

        conv_out_len = calcOut(sequence_len, 10)
        conv_out_len = calcOut(conv_out_len, 1)
        conv_out_len = calcOut(conv_out_len, 3, stride=3)
        conv_out_len = calcOut(conv_out_len, 10)
        conv_out_len = calcOut(conv_out_len, 3, stride=3)
        conv_out_len = calcOut(conv_out_len, 10)
        conv_out_len = calcOut(conv_out_len, 3, stride=3)
        conv_out_len = calcOut(conv_out_len, 10)
        self.conv_out_len = calcOut(conv_out_len, 3, stride=3)

        self.linear_size = (200 * self.conv_out_len)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10)),
            nn.Conv2d(25, 25, kernel_size=(65, 1)),
            nn.BatchNorm2d(25), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(0.3),
            nn.Conv2d(25, 50, kernel_size=(1, 10)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(0.3),
            nn.Conv2d(50, 100, kernel_size=(1, 10)), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(0.3),
            nn.Conv2d(100, 200, kernel_size=(1, 10)), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(0.3)
        )

        self.linear = nn.Sequential(
            nn.Linear(self.linear_size + (9 if self.with_table else 0), 128),
            nn.Linear(128, 10),
            nn.Linear(10, num_classes), )

    def forward(self, x, table=None):
        x = self.conv(x)
        x = x.view(-1, self.linear_size)
        if self.with_table:
            x = torch.cat([x, table], dim=-1)
        x = self.linear(x)
        return x


# single train loop over dataloader
def train(dataloader, model, criterion, optimizer, device):
    running_acc, running_loss = 0.0, 0.0
    model.train()
    for X, row, y in tqdm(dataloader):
        X, y, row = X.to(device), y.to(device), row.to(device)
        X = torch.unsqueeze(X, 1)
        y_pred = model(X, row)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_acc += (y_pred.argmax(dim=-1) == y).sum()
        running_loss += loss.item()

    nb_samples = len(dataloader.dataset)
    nb_batches = len(dataloader)

    acc = (100 * running_acc / nb_samples).item()
    loss = running_loss / nb_batches

    print(f"Train accuracy: {acc:>0.1f}% ({int(running_acc)}/{nb_samples}),"
          f"Train loss: {running_loss / nb_batches:>8f}")
    return acc, loss


# single test loop over dataloader
def test(dataloader, model, criterion, device, forPatient=True):
    model.eval()
    acc_history, loss = 0.0, 0.0
    all_y, all_pred = [], []
    with torch.no_grad():
        for X, table, y in dataloader:
            X, y, table = X.to(device), y.to(device), table.to(device)
            X = torch.unsqueeze(X, 1)
            y_pred = model(X, table)

            """
            one batch contains sample of one patient,
            therefore we take the mean of one batch to 
            get the whole prediction for one patient
            """
            if forPatient:
                y_pred = torch.unsqueeze(torch.mean(y_pred, dim=0), 0)
                y = torch.unsqueeze(torch.mean(y.float(), dim=0).long(), 0)

            loss += criterion(y_pred, y).item()
            acc_history += (y_pred.argmax(dim=-1) == y).sum()

            all_pred = all_pred + (y_pred.cpu().numpy().tolist())
            all_y = all_y + (y.cpu().numpy().tolist())

    nb_samples = len(dataloader.dataset)
    nb_batches = len(dataloader)

    acc = (nb_batches if forPatient else nb_samples)
    acc = (100 * acc_history / acc).item()
    loss = loss / nb_batches

    print(f"{('Patient' if forPatient else 'Sample')} Val accuracy: {acc:>0.1f}% ({int(acc_history)}/{(nb_batches if forPatient else nb_samples)}), Val loss: {loss:>8f}")

    return acc, loss, all_y, all_pred


# single inference loop over dataloader
def inference(dataloader, model):
    model.eval()
    acc_history, loss = 0.0, 0.0
    all_y, all_pred = [], []
    with torch.no_grad():
        for X, table, y in dataloader:
            X, y, table = X, y, table
            X = torch.unsqueeze(X, 1)
            y_pred = model(X, table)

            """
            one batch contains sample of one patient,
            therefore we take the mean of one batch to 
            get the whole prediction for one patient
            """
            y_pred = torch.unsqueeze(torch.mean(y_pred, dim=0), 0)
            y = torch.unsqueeze(torch.mean(y.float(), dim=0).long(), 0)
            acc_history += (y_pred.argmax(dim=-1) == y).sum()

            all_pred = all_pred + (y_pred.argmax(dim=-1).numpy().tolist())
            all_y = all_y + (y.cpu().numpy().tolist())

    nb_batches = len(dataloader)

    acc = (100 * acc_history / nb_batches).item()

    print(f"Val accuracy: {acc:>0.1f}% ({int(acc_history)}/{nb_batches})")

    return acc, all_y, all_pred

# model = Model()
# summary = summary(model, (16, 1, 65, 5500), device='cpu')
# print(summary)
