import torch
from torch import nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torchsummary

from src.monlan.modular_agents.tcn.tcn import *
from src.monlan.utils.save_load import *

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
from datetime import datetime

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class CustomDataset(Dataset):
    def __init__(self, sample_generator, step_ids, use_augments):

        self.sample_generator = sample_generator
        self.step_ids = step_ids
        self.use_augments = use_augments

    def __len__(self):
        return len(self.step_ids)

    def __getitem__(self, idx):
        step_id = self.step_ids[idx]
        x_i, y_i = self.sample_generator.get_sample( step_id )

        # Conv 1d
        #x_i = x_i.reshape( (x_i.shape[1], x_i.shape[2]) )

        # Conv 2d
        x_i = x_i.reshape( (1, x_i.shape[1], x_i.shape[2]) )

        #if self.use_augments:
        #    x_i = self.augment_x( x_i, noise_power=0.01, random_replace_power=0.05 )

        x_i = np.float32(x_i)
        y_i = np.float32(y_i)

        return x_i, y_i

    def augment_x(self, x_i, noise_power=0.01, random_replace_power=0.05):
        raw_x_shape = x_i.shape
        if x_i.shape[0] == 1:
            x_i = x_i.reshape((x_i.shape[1], x_i.shape[2]))

        noise = np.random.standard_normal(x_i.shape)
        x_i = noise_power * noise + x_i

        """replace_ids_count = int(random_replace_power * x_i.shape[1])
        random_column_ids = np.random.randint(low=0, high=x_i.shape[1], size=2 * replace_ids_count)
        ids_to_replace_left = random_column_ids[:replace_ids_count]
        ids_to_replace_right = random_column_ids[replace_ids_count:]
        tmp = x_i.copy()
        x_i[:, ids_to_replace_left] = x_i[:, ids_to_replace_right]
        x_i[:, ids_to_replace_right] = tmp[:, ids_to_replace_left]"""

        x_i = x_i.reshape(raw_x_shape)
        return x_i

class BigSupervisedTCNMirrorCloser:
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

    def fit(self, sample_generator, n_jobs=0, val_size=0.03, val_batch_size=256,
            epochs = 202, warm_up_epochs = 2, batch_size = 256, learning_rate = 0.001, batch_norm_momentum=0.1,
            checkpoint_save_path = os.path.join("../../models", "best_val_big_supervised_tcn_mirror_closer.pkl"),
            continue_train = True):

        def train(dataloader, model, loss_fn, optimizer):
            size = len(dataloader.dataset)
            model.train()
            i = 0
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            start_time = datetime.now()
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x)

                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i += 1
                if i % 100 == 0:
                    total_time = datetime.now() - start_time
                    loss, current = loss.item(), i * len(x)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] {total_time}")

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

        train_steps, val_steps, train_y, val_y = sample_generator.get_train_val_split( test_size=val_size, shuffle=True )
        train_dataset = CustomDataset(sample_generator, train_steps, use_augments=True)
        val_dataset = CustomDataset(sample_generator, val_steps, use_augments=False)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_jobs)
        val_data_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=n_jobs)
        loss_function = torch.nn.MSELoss()


        if continue_train:
            model = load( checkpoint_save_path ).model
        else:
            model = self.build_model(batch_norm_momentum)
        model = model.to(self.device)

        ###########
        # model summaries
        # conv 1d
        #torchsummary.summary(model, (7, 256))

        # conv 2d
        #torchsummary.summary(model, (1, 7, 256))
        ###########

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

    def get_action_head_only(self, state):

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
            state = state.to(self.device)

        with torch.no_grad():
            if isinstance(self.model, torch.nn.Sequential):
                q_value = self.model[-1](state).cpu().detach().numpy()
            else:
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
        #state = state.reshape((1, state.shape[1], state.shape[2]))

        # 2d conv
        state = state.reshape((1, 1, state.shape[1], state.shape[2]))

        state = torch.Tensor(state).to(self.device)

        with torch.no_grad():
            q_value = self.model(state).cpu().detach().numpy()
        ####
        #q_value = np.abs( q_value )
        ####
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
        """model = TCNDenseOutput(input_size=self.input_size,
                               output_size=self.action_size,
                               num_channels=[64] * 16,
                               kernel_size=8,
                               dropout=0.33,
                               batch_norm_momentum=batch_norm_momentum
                               )"""

        model = models.resnet18(pretrained=True, progress=True)
        #model = models.resnet34(pretrained=True, progress=True)
        #model = models.resnet50(pretrained=True, progress=True)
        #model = models.resnet101(pretrained=True, progress=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

        #model = models.regnet_x_3_2gf(pretrained=True, progress=True)
        #model.stem[0] = nn.Conv2d( 1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False )
        #nn.init.kaiming_normal_(model.stem[0].weight, mode="fan_out", nonlinearity="relu")

        #model = models.efficientnet_b1(pretrained=True, progress=True)
        #model.features[0][0] = nn.Conv2d( 1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False )

        embedding_layer = nn.Linear(in_features=1000, out_features=100)
        nn.init.kaiming_normal_(embedding_layer.weight, mode="fan_out", nonlinearity="linear")
        action_layer = nn.Linear(in_features=100, out_features=self.action_size)
        nn.init.kaiming_normal_(action_layer.weight, mode="fan_out", nonlinearity="linear")
        model = nn.Sequential(
            model,
            embedding_layer,
            action_layer
        )

        return model