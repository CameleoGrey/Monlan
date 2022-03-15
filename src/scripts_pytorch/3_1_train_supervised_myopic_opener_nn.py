
################################################################
# Training supervised opener on the dataset from previous step.
################################################################

import os
import numpy as np
from src.monlan.utils.save_load import *
import torch
import torchvision.models
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = np.float32(x)
        self.y = np.float32(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_i = self.x[idx]
        y_i = self.y[idx]
        return x_i, y_i

train_dataset = load( os.path.join("../../data/processed", "train_dataset.pkl") )
test_dataset = load( os.path.join("../../data/processed", "test_dataset.pkl") )
val_dataset = load( os.path.join("../../data/processed", "val_dataset.pkl") )

train_dataset = CustomDataset( train_dataset["x"], train_dataset["y"] )
test_dataset = CustomDataset( test_dataset["x"], test_dataset["y"] )
val_dataset = CustomDataset( val_dataset["x"], val_dataset["y"] )

#model = torchvision.models.resnet18(pretrained=False, progress=True, num_classes=2)
model = torchvision.models.resnet34(pretrained=False, progress=True, num_classes=2)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model = torch.nn.Sequential( model,
                             torch.nn.Tanh()
                             )
model.to( 'cuda' if torch.cuda.is_available() else 'cpu' )

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    i = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for x, y in dataloader:
        x, y = x.to( device ), y.to( device )
        pred = model(x)

        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i += 1
        if i % 100 == 0:
            loss, current = loss.item(), i * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to( device )
            y = y.to( device )
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return test_loss

epochs = 5
batch_size = 16
learning_rate = 0.001

#loss_function = torch.nn.MSELoss()
loss_function = torch.nn.SmoothL1Loss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

best_loss = np.inf
for i in range( epochs ):
    print("Epoch: {}".format(i))
    train( train_data_loader, model, loss_function, optimizer )
    current_loss = test( val_data_loader, model, loss_function )
    print("Validation loss: {}".format(current_loss))
    if current_loss < best_loss:
        print("Previous best loss: {}".format(best_loss))
        best_loss = current_loss
        save( model, os.path.join( "../../models", "supervised_opener_network.pkl" ) )
print("Test loss: ", end="")
test( test_data_loader, model, loss_function )
print("done")