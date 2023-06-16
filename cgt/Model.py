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
    def __init__(self, input_len, output_len, sensors, sensor_dim=1,d_model=32, individual=False):
        super(Model, self).__init__()
        self.embedding=LSTMEmbedding(d_model,sensors=sensors,individual=individual)

    def forward(self, x, graph):
        return self.embedding(x)


if __name__ == '__main__':
    batch_size=10
    input_len=100
    output_len=50
    sensors=5
    a = torch.zeros([batch_size, input_len, sensors])
    output=Model(input_len,output_len,sensors,individual=False)(a,None)
    print(output.shape)
