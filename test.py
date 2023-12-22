import torch

a=[1,2,3,4]
a=torch.tensor(a)
b=torch.fft.fft(a)
c=torch.fft.fft(a)*2
print(b*c)
print(torch.norm(b-c))
print(b-c)
print(torch.fft.ifft(b*c))
print(torch.fft.ifft(b))