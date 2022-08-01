import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from src.classes.utils import *
from src.classes.paths_config import *
from src.classes.tcn.tcn import TCNDenseOutput
from sklearn.metrics import roc_auc_score, accuracy_score

class TrainDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        x = self.data[index]
        x = torch.tensor(x, dtype=torch.float)
        x = x.unsqueeze(0)


        y = self.label[index]
        y = torch.tensor(y, dtype=torch.long)

        return (x, y)

    def __len__(self):
        return len(self.data)

class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]
        x = torch.tensor(x, dtype=torch.float)
        x = x.unsqueeze(0)
        return x

    def __len__(self):
        return len(self.data)

class TCNClassifier():
    def __init__(self, input_channels, num_classes, num_channels=[25]*7, kernel_size=7, dropout=0.05, device="cuda",
                 model_name="pretrained_base_nn_classifier",
                 pretrained_model = True, scale_val=100):

        self.device = device
        self.input_size = input_channels
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.model_name = model_name
        self.pretrained_model = pretrained_model
        self.scale_val = scale_val

        self.base_model = TCNDenseOutput(input_channels, output_size=num_classes,
                                         num_channels=self.num_channels, kernel_size=self.kernel_size,
                                         dropout=self.dropout)
        self.composite_model = self.base_model
        self.composite_model.to(self.device)

        pass

    def fit(self, train_x, train_y, val_x, val_y, batch_size=64, epochs = 200, learning_rate = 0.001):

        def train_epoch(train_data_loader, loss_function, optimizer):
            size = len(train_data_loader.dataset)
            self.composite_model.train()

            i = 0
            epoch_start = datetime.now()
            for x, y in train_data_loader:

                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.composite_model(x)
                loss = loss_function(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i += 1
                if i % 20 == 0:
                    loss, current = loss.item(), i * len(x)
                    print("loss: {:.5}  [{}/{}] {}".format( loss, current, size, datetime.now() - epoch_start ))
            epoch_end = datetime.now()
            print("Total epoch time: {}".format( epoch_end - epoch_start ))

        uniq_y = np.unique( train_y )
        class_counts = []
        for y_u in uniq_y:
            class_count = len(train_y[ train_y == y_u ])
            class_counts.append( class_count )
        class_weights = []
        for cl_c in class_counts:
            cl_w = len( train_y ) / cl_c
            class_weights.append( cl_w )
        class_weights = np.array( class_weights )
        class_weights /= np.sum( class_weights )

        weights = [class_weights[int(train_y[i])] for i in range(len(train_y))]
        sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len((train_y)))

        train_dataset = TrainDataset(train_x, train_y)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        #train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        loss_function = torch.nn.NLLLoss()
        #optimizer = torch.optim.AdamW(self.composite_model.parameters(), lr=learning_rate, weight_decay=1e-2,
        #                              betas=(0.9, 0.999))
        optimizer = torch.optim.Adam(self.composite_model.parameters(), lr=learning_rate)

        best_score = -np.inf
        best_base_model = None
        best_composite_model = None
        for i in range(epochs):
            print("Epoch: {}".format(i))
            train_epoch(train_data_loader, loss_function, optimizer)
            current_score = self.evaluate(val_x, val_y, batch_size=batch_size)
            print("Validation roc_auc: {}".format(current_score))
            if current_score > best_score:
                print("Previous best roc_auc: {}".format(best_score))
                best_score = current_score
                best_base_model = self.base_model
                best_composite_model = self.composite_model
        self.base_model = best_base_model
        self.composite_model = best_composite_model

        pass

    def evaluate(self, val_x, val_y, batch_size):

        val_dataset = TrainDataset(val_x, val_y)
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        y_true = []
        with torch.no_grad():
            for x, y in val_data_loader:
                y_true.append(y)
        y_true = np.hstack(y_true)

        #y_pred = self.predict( val_x, batch_size=batch_size)
        #acc_score = accuracy_score(y_true, y_pred)

        ##################
        #for contest only
        y_pred_proba = self.predict_proba(val_x, batch_size=batch_size)[:, 1]
        roc_auc = roc_auc_score(y_true, y_pred_proba, average=None)
        ##################

        return roc_auc

    def predict_proba(self, x, batch_size=256):

        test_dataset = TestDataset(x)
        test_dataloader = DataLoader( test_dataset, batch_size=batch_size, shuffle=False )

        self.composite_model.eval()

        y_pred = []
        with torch.no_grad():
            for x in test_dataloader:
                x = x.to(self.device)
                pred = self.composite_model(x)
                pred = torch.exp(pred)
                pred = pred.to("cpu").detach().numpy()

                y_pred.append(pred)

        y_pred = np.vstack(y_pred)

        return y_pred

    def predict(self, x, batch_size):

        y_pred = self.predict_proba( x, batch_size=batch_size )

        predicts = []
        for i in range(y_pred.shape[0]):
            """for j in range(y_pred.shape[1]):
                if y_pred[i][j] > 0.5:
                    y_pred[i][j] = 1.0
                else:
                    y_pred[i][j] = 0.0"""
            y_pred_i = np.argmax( y_pred[i] )
            predicts.append( y_pred_i )
        y_pred = np.array( predicts, dtype=np.int )

        return y_pred

    def get_embeddings(self, x, batch_size):

        test_dataset = TestDataset( x )
        test_dataloader = DataLoader( test_dataset, batch_size=batch_size, shuffle=False )

        self.composite_model.eval()

        y_pred = []
        with torch.no_grad():
            for x in tqdm(test_dataloader, desc="making embeddings"):
                x = x.to(self.device)
                pred = self.base_model(x)
                pred = torch.flatten( pred, 1 )

                pred = pred.to("cpu").detach().numpy()
                y_pred.append( pred )

        y_pred = np.vstack(y_pred)

        return y_pred
