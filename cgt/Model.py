import torch
from torch import nn


class LSTMEmbedding(nn.Module):
    def __init__(self, output_channels, sensors=1, input_size=1, individual=False):
        super(LSTMEmbedding, self).__init__()
        if individual:
            self.lstm = nn.ModuleList(
                [nn.LSTM(input_size=input_size, hidden_size=output_channels, num_layers=1, batch_first=True) for _ in
                 range(sensors)])
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=output_channels, num_layers=1, batch_first=True)

    def forward(self, x):
        batch_size, input_len, sensors = x.shape
        print(batch_size, input_len, sensors)
        if isinstance(self.lstm, nn.ModuleList):
            return torch.cat([lstm(x[:, :, i].unsqueeze(2))[0] for i, lstm in enumerate(self.lstm)]).reshape(batch_size,
                                                                                                             sensors,
                                                                                                             input_len,
                                                                                                             -1)
        else:
            x = x.transpose(1, 2)
            x = x.reshape(x.shape[0] * x.shape[1], -1, 1)
            # print(x.shape)
            return self.lstm(x)[0].reshape(batch_size, sensors, input_len, -1)


class Model(nn.Module):
    def __init__(self, input_len, output_len, sensors, sensor_dim, individual=False):
        super(Model, self).__init__()

    def forward(self, x):
        pass


if __name__ == '__main__':
    a = torch.zeros([10, 100, 5])
    embedding = LSTMEmbedding(32, 5, 1, True)
    print(embedding(a).shape)
