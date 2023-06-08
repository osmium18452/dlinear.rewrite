import torch
from torch import nn


class LTSFLinear(nn.Module):
    def __init__(self, input_size, output_size, sensors=0, individual=False):
        super(LTSFLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sensors = sensors
        self.individual = individual
        self.linear = None
        if individual:
            self.linear = nn.ModuleList()
            for i in range(sensors):
                self.linear.append(nn.Linear(input_size, output_size))
        else:
            self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        if self.individual:
            output = torch.zeros([x.size(0), self.output_size, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.sensors):
                output[:, :, i] = self.linear[i](x[:, :, i])
            x = output
        else:
            x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class LTSFNLinear(nn.Module):
    def __init__(self, input_size, output_size, sensors=0, individual=False):
        super(LTSFNLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sensors = sensors
        self.individual = individual
        self.linear = None
        if individual:
            self.linear = nn.ModuleList()
            for i in range(sensors):
                self.linear.append(nn.Linear(input_size, output_size))
        else:
            self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.output_size, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.sensors):
                output[:, :, i] = self.linear[i](x[:, :, i])
            x = output
        else:
            x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class LTSFDLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, input_size,output_size,sensors=0,individual=False,kernel_size=25):
        super(LTSFDLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.decompsition = SeriesDecomp(kernel_size)
        self.individual = individual
        self.sensors = sensors

        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()

            for i in range(self.channels):
                self.linear_seasonal.append(nn.Linear(self.input_size, self.output_size))
                self.linear_trend.append(nn.Linear(self.input_size, self.output_size))
        else:
            self.linear_seasonal = nn.Linear(self.input_size, self.output_size)
            self.linear_trend = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.output_size], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.output_size], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:,i,:] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]