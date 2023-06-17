import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model=128, n_heads=8, input_size=1, dropout=0.1):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.q_list = nn.ModuleList(
            [nn.LSTM(input_size=input_size, hidden_size=d_model, num_layers=1, batch_first=True) for _ in
             range(n_heads)])
        self.k_list = nn.ModuleList(
            [nn.LSTM(input_size=input_size, hidden_size=d_model, num_layers=1, batch_first=True) for _ in
             range(n_heads)])
        self.v_list = nn.ModuleList(
            [nn.LSTM(input_size=input_size, hidden_size=1, num_layers=1, batch_first=True) for _ in range(n_heads)])
        # self.q_lstm = nn.LSTM(input_size=input_size, hidden_size=d_model, num_layers=1, batch_first=True)
        # self.k_lstm = nn.LSTM(input_size=input_size, hidden_size=d_model, num_layers=1, batch_first=True)
        # self.v_lstm = nn.LSTM(input_size=input_size, hidden_size=1, num_layers=1, batch_first=True)
        self.leaky_relu = nn.LeakyReLU(.2)

    def forward(self, x, graph):
        """

        :param x: time series, [batch_size, sensors, input_len, 1]
        :param graph: adjacency matrix
        :return: [batch_size, sensors, input_len, n_heads]
        """
        output_list = []
        graph = graph.transpose(0, 1) + torch.eye(graph.shape[0]).to(graph.device)
        mask = torch.where(graph == 0, -1e9, 0)
        ori_shape = x.shape
        x = x.reshape(ori_shape[0] * ori_shape[1], -1, 1)
        for i in range(self.n_heads):
            _, (q, __) = self.q_list[i](x)
            _, (k, __) = self.k_list[i](x)
            v, (_, __) = self.v_list[i](x)
            q = q.reshape(ori_shape[0], ori_shape[1], -1)
            k = k.reshape(ori_shape[0], ori_shape[1], -1)
            v = v.reshape(ori_shape[0], ori_shape[1], -1)
            attention = torch.matmul(q, k.transpose(1, 2)) / self.d_model ** 0.5
            attention = self.leaky_relu(attention)
            masked_attention = F.softmax(attention + mask, dim=-1)
            # print(masked_attention, v.shape)
            output = torch.matmul(masked_attention, v)
            output_list.append(torch.unsqueeze(output, -1))
        # print(.shape)
        return self.leaky_relu(torch.cat(output_list, -1))


class Model(nn.Module):
    def __init__(self, input_len, output_len, sensors, d_model=128, n_heads=8, individual=False):
        super(Model, self).__init__()
        self.individual = individual
        self.sensors = sensors
        self.attn = Attention(d_model=d_model, n_heads=n_heads)
        self.relu = nn.ReLU()
        self.activation_func = nn.LeakyReLU(.2)
        if individual:
            self.dim_list = nn.ModuleList([nn.Linear(n_heads, 1) for _ in range(sensors)])
            self.time_list = nn.ModuleList([nn.Linear(input_len, output_len) for _ in range(sensors)])
        else:
            self.dim_linear = nn.Linear(n_heads, 1)
            self.time_linear = nn.Linear(input_len, output_len)

    def forward(self, input_x, encoding_x, input_y, encoding_y, graph):
        input_x = torch.unsqueeze(input_x.transpose(1, 2), dim=-1)
        h = self.attn(input_x, graph)
        if self.individual:
            h_list = []
            for i in range(self.sensors):
                tmp = torch.squeeze(self.dim_list[i](h[:, i, :, :] + input_x[:, i, :, :]), dim=-1)
                tmp = self.activation_func(tmp)
                h_list.append(torch.unsqueeze(self.time_list[i](tmp), dim=1))
            h = torch.cat(h_list, dim=1)
        else:
            h = torch.squeeze(self.dim_linear(h) + input_x, dim=-1)
            h = self.activation_func(h)
            h = self.time_linear(h)
        return h.transpose(1, 2)


if __name__ == '__main__':
    attn = Attention()
    batch_size = 2
    sensors = 5
    input_len = 50
    output_len = 24
    x = torch.zeros([batch_size, input_len, sensors])
    graph = torch.zeros([sensors, sensors])
    model = Model(input_len, output_len, sensors, individual=True)
    # y=model(x, graph)
    # print(y.shape)
