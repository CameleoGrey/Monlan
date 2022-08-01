

import os
import numpy as np
from src.monlan.utils.save_load import *
import torch
import torchvision.models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

def get_embeddings(dataloader, model):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = []
    targets = []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extracting embeddings"):
            x = x.to( device )
            pred = model.get_embeddings(x)
            pred = pred.cpu().detach().numpy()
            embeddings.append( pred )
            targets.append( y )
    embeddings = np.vstack( embeddings )
    targets = np.vstack( targets )

    return embeddings, targets


train_data_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
val_data_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

model = load( os.path.join( "../../models", "supervised_opener_network.pkl" ) )
model = model._modules["0"]
#modules = list(model.children())[:-1]
#model = torch.nn.Sequential(*modules)
model.cuda()

train_x, train_y = get_embeddings(train_data_loader, model)
val_x, val_y = get_embeddings(val_data_loader, model)
test_x, test_y = get_embeddings(test_data_loader, model)

train_dataset = {"x": train_x, "y": train_y}
val_dataset = {"x": val_x, "y": val_y}
test_dataset = {"x": test_x, "y": test_y}

save( train_dataset, os.path.join("../../data/processed", "train_vector_dataset.pkl") )
save( val_dataset, os.path.join("../../data/processed", "val_vector_dataset.pkl") )
save( test_dataset, os.path.join("../../data/processed", "test_vector_dataset.pkl") )

print("done")
