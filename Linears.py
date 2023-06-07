import torch
from torch import nn

class LSTFLinear(nn.Module):
    def __init__(self,input_size,output_size,sensors=0,individual=False):
        super(LSTFLinear, self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.sensors=sensors
        self.individual=individual
        self.linear=None
        if individual:
            self.linear=nn.ModuleList()
            for i in range(sensors):
                self.linear.append(nn.Linear(input_size,output_size))
        else:
            self.linear=nn.Linear(input_size,output_size)

    def forward(self,x):
        if self.individual:
            output = torch.zeros([x.size(0),self.output_size,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.sensors):
                output[:,:,i] = self.linear[i](x[:,:,i])
            x = output
        else:
            x = self.linear(x.permute(0,2,1)).permute(0,2,1)
        return x