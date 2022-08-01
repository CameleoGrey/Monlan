import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset

from src.monlan.modular_agents.tcn.tcn import *
from src.monlan.utils.save_load import *

from datetime import datetime

import os
import random
import numpy as np
from collections import deque
from scipy.special import softmax
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = np.float32(x)
        self.y = np.float32(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_i = self.x[idx]
        x_i = x_i.reshape( (x_i.shape[1], x_i.shape[2]) )
        y_i = self.y[idx]
        return x_i, y_i

class DistilledTCNMirrorCloser:
    def __init__(self, input_size):

        self.name = None
        self.input_size = input_size

        self.state_size = None
        self.action_size = 2
        self.epsilon = 0.0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def set_name(self, name):
        self.name = name
        pass

    def get_name(self):
        return self.name

    def fit(self, buyer_samples, seller_samples,
            epochs = 202, warm_up_epochs = 2, batch_size = 16, learning_rate = 0.0001, batch_norm_momentum=0.00625,
            checkpoint_save_path = os.path.join("../../models", "best_val_distilled_tcn_mirror_closer.pkl")):

        def train(dataloader, model, loss_fn, optimizer):
            size = len(dataloader.dataset)
            model.train()
            i = 0
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
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
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)
                    test_loss += loss_fn(pred, y).item()
            test_loss /= num_batches
            return test_loss

        train_x, val_x, train_y, val_y = self.build_train_val_dataset_(buyer_samples, seller_samples)
        train_dataset = CustomDataset(train_x, train_y)
        val_dataset = CustomDataset(val_x, val_y)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        model = self.build_model(batch_norm_momentum).to(self.device)
        loss_function = torch.nn.MSELoss()

        best_loss = np.inf
        for i in range(epochs):
            print("Epoch: {}".format(i))

            if i < warm_up_epochs:
                optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-6, weight_decay=1e-2, betas=(0.9, 0.999),
                                              amsgrad=True)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2,
                                              betas=(0.9, 0.999), amsgrad=True)

            train(train_data_loader, model, loss_function, optimizer)
            val_loss = test(val_data_loader, model, loss_function)
            print("Validation loss: {}".format(val_loss))
            if val_loss < best_loss:
                print("Previous best loss: {}".format(best_loss))
                best_loss = val_loss
                self.model = model
                save(self, checkpoint_save_path)

        best_agent = load( checkpoint_save_path )
        self.model = best_agent.model

        return self

    def build_train_val_dataset_(self, buyer_samples, seller_samples):

        buyer_steps = buyer_samples["id"]
        buyer_x = buyer_samples["x_raw"]
        buyer_y = buyer_samples["y"]

        seller_steps = seller_samples["id"]
        seller_x = seller_samples["x_raw"]
        seller_y = seller_samples["y"]

        common_x = []
        common_y = []
        min_dataset_len = min(len(buyer_steps), len(seller_steps))
        for i in tqdm(range(min_dataset_len), desc="Building common dataset"):
            if buyer_steps[i] != seller_steps[i]:
                raise ValueError("Different step_i")
            if buyer_x[i][0][0][0][0] != seller_x[i][0][0][0][0]:
                raise ValueError("Different obs")

            target = [buyer_y[i], seller_y[i]]
            target_sum = np.sum(target)
            if np.isnan( target_sum ):
                continue
            common_y.append(target)
            common_x.append(buyer_x[i])

        common_x = np.array(common_x)
        common_y = np.array(common_y)

        common_x = common_x.reshape((common_x.shape[0], common_x.shape[1], common_x.shape[2], common_x.shape[3]))

        x = common_x
        y = common_y

        x, y = self.ema_normalize_targets_(x, y, n=20)
        x, y = self.smooth_targets_( x, y, n=20 )

        #axis_x = [i for i in range(len(y))]
        #plt.plot( axis_x, y[:, 0], c="g" )
        #plt.plot( axis_x, y[:, 1], c="r" )
        #plt.show()

        # targets for scaler without outliers
        buyer_quantile = np.quantile(np.abs(y[:, 0]), q=0.99)
        seller_quantile = np.quantile(np.abs(y[:, 1]), q=0.99)
        scale_y_buyer = y[np.abs(y[:, 0]) < buyer_quantile, 0]
        scale_y_seller = y[np.abs(y[:, 1]) < seller_quantile, 1]
        scale_y = np.hstack( [scale_y_buyer, scale_y_seller] )
        scale_y = scale_y.reshape((scale_y.shape[-1], 1))

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(scale_y)
        y[:, 0] = scaler.transform(y[:, 0].reshape((-1, 1))).reshape((-1,))
        y[:, 1] = scaler.transform(y[:, 1].reshape((-1, 1))).reshape((-1,))

        # remove close to zero samples
        #x = x[(y[:, 0] < -0.25) | (y[:, 0] > 0.05) ]
        #y = y[(y[:, 0] < -0.25) | (y[:, 0] > 0.05)]
        #x = x[(y[:, 1] < -0.25) | (y[:, 1] > 0.05)]
        #y = y[(y[:, 1] < -0.25) | (y[:, 1] > 0.05)]

        sns.displot( y.reshape((y.shape[0] * y.shape[1], 1)) )
        plt.show()

        train_x, val_x, train_y, val_y = train_test_split(x, y, shuffle=True, test_size=0.1, random_state=45)

        print("Train dataset size: {}".format( len(train_y) ))
        print("Val dataset size: {}".format(len(val_y)))

        return train_x, val_x, train_y, val_y

    def ema_normalize_targets_(self, x, y, n=40):
        alpha = 1.0 / float(n)

        buyer_ema = sum(y[:n, 0]) / float(n)
        seller_ema = sum(y[:n, 1]) / float(n)

        x = x[n:]
        y = y[n:]
        for i in range(len(y)):
            y[i][0] = y[i][0] / (np.abs(buyer_ema) + 1.0)
            y[i][1] = y[i][1] / (np.abs(seller_ema) + 1.0)
            buyer_ema = alpha * y[i][0] + (1.0 - alpha) * buyer_ema
            seller_ema = alpha * y[i][1] + (1.0 - alpha) * seller_ema

        return x, y

    def smooth_targets_(self, x, y, n=40):

        alpha = 1.0 / float(n)

        buyer_ema = sum( y[:n, 0] ) / float( n )
        seller_ema = sum( y[:n, 1] ) / float( n )

        x = x[n:]
        y = y[n:]
        for i in range( len(y) ):
            buyer_ema = alpha * y[i][0] + (1.0 - alpha) * buyer_ema
            seller_ema = alpha * y[i][1] + (1.0 - alpha) * seller_ema
            y[i][0] = buyer_ema
            y[i][1] = seller_ema

        return x, y


    def get_action_head_only(self, state):

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
            state = state.to(self.device)

        with torch.no_grad():
            q_value = self.model.linear(state).cpu().detach().numpy()
        if self.name == "buyer":
            q_value[0], q_value[1] = q_value[0], q_value[1]
        elif self.name == "seller":
            q_value[1], q_value[0] = q_value[0], q_value[1]
        else:
            raise ValueError("Name of mirror agent must be specified (buyer, seller) before using")

        greed_action_id = np.argmax(q_value)

        return greed_action_id

    def get_action(self, state):

        # 1d conv
        state = state.reshape((1, state.shape[1], state.shape[2]))

        # 2d conv
        # state = state.reshape((1, 1, state.shape[1], state.shape[2]))

        state = torch.Tensor(state).to(self.device)

        with torch.no_grad():
            q_value = self.model(state).cpu().detach().numpy()
        q_value = q_value[0]
        if self.name == "buyer":
            q_value[0], q_value[1] = q_value[0], q_value[1]
        elif self.name == "seller":
            q_value[1], q_value[0] = q_value[0], q_value[1]
        else:
            raise ValueError("Name of mirror agent must be specified (buyer, seller) before using")

        greed_action_id = np.argmax(q_value)

        # idea: close potentially unprofitable deals anyway
        #if np.amax( q_value ) < 0.05:
        #    greed_action_id = 1

        return greed_action_id

    def build_model(self, batch_norm_momentum):
        model = TCNDenseOutput(input_size=self.input_size,
                               output_size=self.action_size,
                               num_channels=[64] * 16,
                               kernel_size=8,
                               dropout=0.33,
                               batch_norm_momentum=batch_norm_momentum
                               )
        return model