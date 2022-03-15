

from torch.autograd import Variable
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np

class ModelLSTM(nn.Module):

    def __init__(self, num_classes=1, input_size=128, hidden_size=512, num_layers=4, use_device="cuda"):

        super(ModelLSTM, self).__init__()

        self.use_device = use_device
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.batch_size = 1
        # self.seq_length = seq_length

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(256, 128)

        self.bn2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dp2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()
        self.to( self.use_device )

    def forward(self, x):
        h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.use_device))
        c_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.use_device))

        _, (hn, cn) = self.lstm_layer(x, (h_1, c_1))

        # print("hidden state shpe is:",hn.size())
        y = hn.view(-1, self.hidden_size)

        final_state = hn.view(self.num_layers, x.size(0), self.hidden_size)[-1]
        # print("final state shape is:",final_state.shape)

        """x0 = self.fc1(final_state)
        x0 = self.bn1(x0)
        #x0 = self.dp1(x0)
        x0 = self.tanh(x0)

        x0 = self.fc2(x0)
        x0 = self.bn2(x0)
        #x0 = self.dp2(x0)

        x0 = self.tanh(x0)

        out = self.fc3(x0)"""

        out = self.fc3( final_state )
        # print(out.size())
        return out

    def fit(self, x_train, y_train, x_val, y_val, epochs=500, learning_rate=1e-3):

        x_train = np.expand_dims(x_train, axis=1)
        x_val = np.expand_dims(x_val, axis=1)

        x_train = torch.Tensor(x_train).to(self.use_device)
        y_train = torch.Tensor(y_train).to(self.use_device)
        x_val = torch.Tensor(x_val).to(self.use_device)
        y_val = torch.Tensor(y_val).to(self.use_device)

        best_val_loss = 10000000

        self.apply(self.init_weights)

        criterion = torch.nn.MSELoss().to(self.use_device)  # mean-squared error for regression
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Train the model
        for epoch in tqdm(range(epochs)):
            self.train()
            outputs = self(x_train)
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            # obtain the loss function
            loss = criterion(outputs, y_train)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(lstm.parameters(), 1)

            optimizer.step()
            self.eval()
            valid = self(x_val)
            vall_loss = criterion(valid, y_val)

            if epoch % 1 == 0:
                print("Epoch: %d, loss: %1.5f valid loss:  %1.5f " % (epoch, loss.cpu().item(), vall_loss.cpu().item()))

    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def predict(self, x):
        self.eval()
        x = np.expand_dims(x, axis=1)
        x = torch.Tensor(x).to(self.use_device)
        pred = self.forward(x)
        pred = pred.detach().numpy().reshape((-1,))

        return pred