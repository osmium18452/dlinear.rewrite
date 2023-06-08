import os
import pickle
import argparse
import platform
import random

import torch
from torch.utils.data import DataLoader
from Datapreprocessor import Datapreprocessor, LinearsDataset, InformerDataset
from Linears import LTSFLinear, LTSFNLinear, LTSFDLinear
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=1024)
parser.add_argument('-B', '--best_model', action='store_true')
parser.add_argument('-C', '--CUDA_VISIBLE_DEVICES', type=str, default='0,1,2,3,4,5,6,7')
parser.add_argument('-d', '--dataset', type=str, default='gweather', help='wht, gweather')
parser.add_argument('-D', '--delete_model_dic', action='store_true')
parser.add_argument('-e', '--total_eopchs', type=int, default=100)
parser.add_argument('-E', '--early_stop', action='store_true')
parser.add_argument('-f', '--fixed_seed', type=int, default=None)
parser.add_argument('-G', '--gpu', action='store_true')
parser.add_argument('-I', '--individual', action='store_true')
parser.add_argument('-i', '--input_len', type=int, default=336)
parser.add_argument('-k', '--kernel_size', type=int, default=25)
parser.add_argument('-l', '--lr', type=float, default=.001)
parser.add_argument('-m', '--model', type=str, default='linear', help='linear, dlinear, nlinear')
parser.add_argument('-o', '--output_len', type=int, default=96)
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
fixed_seed = args.fixed_seed
best_model = args.best_model
kernel_size = args.kernel_size
delete_model_dic = args.delete_model_dic
early_stop = args.early_stop

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
device = torch.device('cuda:0' if gpu else 'cpu')
# print(device)

if fixed_seed is not None:
    random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)

if platform.system() == 'Windows':
    data_root = 'E:\\forecastdataset\\pkl'
else:
    data_root = '/home/icpc/pycharmproj/forecast.dataset/pkl/'

data_preprocessor = None
dataset = None
if dataset_name == 'wht':
    dataset = pickle.load(open(os.path.join(data_root, 'wht.pkl'), 'rb'))
elif dataset_name == 'gweather':
    dataset = pickle.load(open(os.path.join(data_root, 'weather.pkl'), 'rb'))
else:
    print('\033[32mno such dataset\033[0m')
    exit()

data_preprocessor = Datapreprocessor(dataset, input_len, output_len, stride=stride)
num_sensors = data_preprocessor.num_sensors
train_input, train_gt = data_preprocessor.load_train_samples()
valid_input, valid_gt = data_preprocessor.load_validate_samples()
test_input, test_gt = data_preprocessor.load_test_samples()
train_loader = DataLoader(LinearsDataset(train_input, train_gt), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(LinearsDataset(valid_input, valid_gt), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(LinearsDataset(test_input, test_gt), batch_size=batch_size, shuffle=False)

model = None
if which_model == 'linear':
    model = LTSFLinear(input_size=input_len, output_size=output_len, sensors=num_sensors, individual=individual)
elif which_model == 'nlinear':
    model = LTSFNLinear(input_size=input_len, output_size=output_len, sensors=num_sensors, individual=individual)
elif which_model == 'dlinear':
    model = LTSFDLinear(input_size=input_len, output_size=output_len, sensors=num_sensors, individual=individual,
                        kernel_size=kernel_size)
else:
    print('\033[32mno such model\033[0m')
    exit()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

pbar_epoch = tqdm(total=total_eopchs, ascii=True, dynamic_ncols=True)
minium_loss = 100000
validate_loss_list = []
last_save_step = -1
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
    with torch.no_grad():
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
        last_save_step = epoch
        minium_loss = validate_loss
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        pbar_epoch.set_description_str('saved at epoch %d %.4f' % (epoch + 1, minium_loss))
    if early_stop and epoch - last_save_step > 10:
        break
pbar_epoch.close()

# test
pbar_iter = tqdm(total=len(test_loader), ascii=True, dynamic_ncols=True)
pbar_iter.set_description_str('testing')
output_list = []
gt_list = []
if best_model:
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
model.eval()
with torch.no_grad():
    for i, (input, ground_truth) in enumerate(test_loader):
        input = input.to(device)
        output = model(input)
        output_list.append(output.cpu())
        gt_list.append(ground_truth)
        pbar_iter.update(1)
pbar_iter.close()
output_list = torch.concatenate(output_list, dim=0)
gt_list = torch.concatenate(gt_list, dim=0)
test_loss = loss_fn(output_list, gt_list).item()
mae_loss = torch.mean(torch.abs(output_list - gt_list))
print('\033[32mmse loss:{:.4f} mae loss:{:.4f}\033[0m'.format(test_loss, mae_loss))
if delete_model_dic:
    os.remove(os.path.join(save_dir, 'best_model.pth'))
    print('\033[33mdeleted model.pth\033[0m')
