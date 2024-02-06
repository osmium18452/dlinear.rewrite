import matplotlib.pyplot as plt
import pickle

import numpy as np

data_file="E:\\forecastdataset\\pkl\\ETTh1.pkl"

f=open(data_file,'rb')
data=pickle.load(f).transpose()[:5,:1000]
print(data.shape)
plt.figure()
plt.imshow(np.repeat(data,100,0))
plt.xticks([])
plt.yticks([])
plt.savefig('freq.png',dpi=300)