import os
import pickle
import argparse
import platform

import torch
from torch.utils.data import DataLoader
from Datapreprocessor import Datapreprocessor, TSDataset
from Linears import LSTFLinear
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=1024)
parser.add_argument('-C', '--CUDA_VISIBLE_DEVICES', type=str, default='0,1,2,3,4,5,6,7')
parser.add_argument('-d', '--dataset', type=str, default='wht')
parser.add_argument('-e', '--total_eopchs', type=int, default=20)
parser.add_argument('-G', '--gpu', action='store_true')
parser.add_argument('-I', '--individual', action='store_true')
parser.add_argument('-i', '--input_len', type=int, default=60)
parser.add_argument('-l', '--lr', type=float, default=.001)
parser.add_argument('-m', '--model', type=str, default='linear', help='linear, dlinear, nlinear')
parser.add_argument('-o', '--output_len', type=int, default=24)
parser.add_argument('-s', '--stride', type=int, default=1)
parser.add_argument('-S', '--save_dir', type=str, default='save')
parser.add_argument('-t', '--train_ratio', type=float, default=.6)
parser.add_argument('-v', '--valid_ratio', type=float, default=.2)
args = parser.parse_args()

dataset_name = args.dataset
input_len = args.input_len
output_len = args.output_len
stride = args.stride
train_ratio = args.train_ratio
valid_ratio = args.valid_ratio
batch_size = args.batch_size
total_eopchs = args.total_eopchs
gpu = args.gpu
individual = args.individual
lr = args.lr
which_model = args.model
save_dir = args.save_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
device = torch.device('cuda:' + args.CUDA_VISIBLE_DEVICES[0] if torch.cuda.is_available() else 'cpu')

print(device)

if platform.system() == 'Windows':
    data_root = 'E:\\forecastdataset\\pkl'
else:
    data_root = '/home/icpc/pycharmproj/forecast.dataset/pkl/'

data_preprocessor = None
dataset = None
if dataset_name == 'wht':
    dataset = pickle.load(open(os.path.join(data_root, 'wht.pkl'), 'rb'))
else:
    print('\033[32mno such dataset\033[0m')
    exit()

data_preprocessor = Datapreprocessor(dataset, input_len, output_len, stride=stride)
num_sensors = data_preprocessor.num_sensors
train_input, train_gt = data_preprocessor.load_train_samples()
valid_input, valid_gt = data_preprocessor.load_validate_samples()
test_input, test_gt = data_preprocessor.load_test_samples()
train_loader = DataLoader(TSDataset(train_input, train_gt), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(TSDataset(valid_input, valid_gt), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TSDataset(test_input, test_gt), batch_size=batch_size, shuffle=False)

model = None
if which_model == 'linear':
    model = LSTFLinear(input_size=input_len, output_size=output_len, sensors=num_sensors, individual=individual)
else:
    print('\033[32mno such model\033[0m')
    exit()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

pbar_epoch = tqdm(total=total_eopchs, ascii=True, dynamic_ncols=True)
minium_loss = 100000
validate_loss_list = []
for epoch in range(total_eopchs):
    # train
    model.train()
    total_iters = len(train_loader)
    pbar_iter = tqdm(total=total_iters, ascii=True, dynamic_ncols=True, leave=False)
    for i, (input, ground_truth) in enumerate(train_loader):
        optimizer.zero_grad()
        input = input.to(device)
        ground_truth = ground_truth.to(device)
        output = model(input)
        loss = loss_fn(output, ground_truth)
        loss.backward()
        optimizer.step()
        pbar_iter.set_postfix_str('loss:{:.4f}'.format(loss.item()))
        pbar_iter.update(1)
    pbar_iter.close()

    # validate
    model.eval()
    output_list = []
    gt_list = []
    pbar_iter = tqdm(total=len(valid_loader), ascii=True, dynamic_ncols=True, leave=False)
    pbar_iter.set_description_str('validating')
    for i, (input, ground_truth) in enumerate(valid_loader):
        input = input.to(device)
        output = model(input)
        output_list.append(output.cpu())
        gt_list.append(ground_truth)
        pbar_iter.update()
    pbar_iter.close()
    output_list = torch.concatenate(output_list, dim=0)
    gt_list = torch.concatenate(gt_list, dim=0)
    validate_loss = loss_fn(output_list, gt_list).item()
    validate_loss_list.append(validate_loss)
    pbar_epoch.set_postfix_str('validate_loss:{:.4f}'.format(validate_loss))
    pbar_epoch.update(1)
    if validate_loss < minium_loss:
        minium_loss = validate_loss
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        pbar_epoch.set_description_str('saved at epoch %d %.4f' % (epoch + 1, minium_loss))
pbar_epoch.close()

# test
pbar_iter = tqdm(total=len(test_loader), ascii=True, dynamic_ncols=True)
pbar_iter.set_description_str('testing')
output_list = []
gt_list = []
for i, (input, ground_truth) in enumerate(test_loader):
    input = input.to(device)
    output = model(input)
    output_list.append(output.cpu())
    gt_list.append(ground_truth)
    pbar_iter.update(1)
pbar_iter.close()
output_list = torch.concatenate(output_list, dim=0)
gt_list = torch.concatenate(gt_list, dim=0)
test_loss = loss_fn(output_list,gt_list).item()
print('\033[32mtest loss:{:.4f}\033[0m'.format(test_loss))
