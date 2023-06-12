import json
import os
import pickle
import argparse
import platform
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from Datapreprocessor import Datapreprocessor, InformerDataset
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-B', '--best_model', action='store_true')
    parser.add_argument('-C', '--CUDA_VISIBLE_DEVICES', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('-d', '--dataset', type=str, default='gweather', help='wht, gweather')
    parser.add_argument('-D', '--delete_model_dic', action='store_true')
    parser.add_argument('-e', '--total_eopchs', type=int, default=20)
    parser.add_argument('-f', '--fixed_seed', type=int, default=None)
    parser.add_argument('-F', '-freq', type=str, default='h', help='time encoding type')
    parser.add_argument('-G', '--gpu', action='store_true')
    parser.add_argument('-I', '--individual', action='store_true')
    parser.add_argument('-i', '--input_len', type=int, default=96)
    parser.add_argument('-k', '--kernel_size', type=int, default=25)
    parser.add_argument('-l', '--lr', type=float, default=.001)
    parser.add_argument('-m', '--model', type=str, default='informer',
                        help='informer, autoformer, fedformer, pyraformer, transformer, reformer')
    parser.add_argument('-M', '--multi_GPU', action='store_true')
    parser.add_argument('-o', '--output_len', type=int, default=96)
    parser.add_argument('-s', '--stride', type=int, default=1)
    parser.add_argument('-S', '--save_dir', type=str, default='save')
    parser.add_argument('-t', '--train_ratio', type=float, default=.6)
    parser.add_argument('-v', '--valid_ratio', type=float, default=.2)
    parser.add_argument('--fudan', action='store_true')
    args = parser.parse_args()
    arg_dict = vars(args)

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
    multiGPU = args.multi_GPU

    local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    if multiGPU:
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cuda:0' if gpu else 'cpu')

    if (multiGPU and local_rank == 0) or not multiGPU:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(json.dumps(arg_dict, ensure_ascii=False))

    if (fixed_seed is not None) or multiGPU:
        seed = fixed_seed if fixed_seed is not None else 2333
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if multiGPU:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")

    if platform.system() == 'Windows':
        data_root = 'E:\\forecastdataset\\pkl'
    else:
        if args.fudan:
            data_root = '/remote-home/liuwenbo/pycproj/dataset'
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

    train_input, train_gt, train_encoding = data_preprocessor.load_train_samples(encoding=True)
    valid_input, valid_gt, valid_encoding = data_preprocessor.load_validate_samples(encoding=True)
    test_input, test_gt, test_encoding = data_preprocessor.load_test_samples(encoding=True)
    train_set = InformerDataset(train_input, train_gt, train_encoding)
    train_loader = DataLoader(train_set, sampler=DistributedSampler(train_set) if multiGPU else None,
                              batch_size=batch_size,
                              shuffle=False if multiGPU else True)
    # valid_loader = None
    test_loader = None
    valid_loader = DataLoader(InformerDataset(valid_input, valid_gt, valid_encoding), batch_size=batch_size,
                              shuffle=False)
    if (multiGPU and local_rank == 0) or not multiGPU:
        test_loader = DataLoader(InformerDataset(test_input, test_gt, test_encoding), batch_size=batch_size,
                                 shuffle=False)

    model = None
    if which_model == 'informer':
        from Informer import Informer

        enc_in = num_sensors
        dec_in = num_sensors
        c_out = num_sensors
        out_len = output_len
        model = Informer(enc_in, dec_in, c_out, out_len)
    elif which_model == 'autoformer':
        from Autoformer import Autoformer

        enc_in = num_sensors
        dec_in = num_sensors
        c_out = num_sensors
        out_len = output_len
        model = Autoformer(enc_in, dec_in, c_out, out_len)
    elif which_model == 'fedformer':
        from FEDformer import Fedformer

        enc_in = num_sensors
        dec_in = num_sensors
        c_out = num_sensors
        out_len = output_len
        model = Fedformer(enc_in, dec_in, c_out, input_len, out_len)
    elif which_model == 'pyraformer':
        from pyraformer import Pyraformer

        enc_in = num_sensors
        out_len = output_len
        input_size = input_len
        model = Pyraformer(enc_in, out_len, input_size, device)
    elif which_model == 'crossformer':
        from Crossformer import Crossformer

        data_dim, in_len, out_len = num_sensors, input_len, output_len
        model = Crossformer(data_dim, in_len, out_len)
    elif which_model == 'transformer':
        from Transformer import Transformer

        input_size = num_sensors
        model = Transformer(input_size)
    elif which_model == 'reformer':
        from Refomer import Reformer

        pred_len, enc_in, c_out = output_len, num_sensors, num_sensors
        model = Reformer(pred_len, enc_in, c_out)
    else:
        print('\033[32mno such model\033[0m')
        exit()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    pbar_epoch = None
    if (multiGPU and local_rank == 0) or not multiGPU:
        pbar_epoch = tqdm(total=total_eopchs, ascii=True, dynamic_ncols=True)
    minium_loss = 100000
    validate_loss_list = []
    last_save_step = -1
    for epoch in range(total_eopchs):
        # train
        model.train()
        total_iters = len(train_loader)
        pbar_iter = None
        if (multiGPU and local_rank == 0) or not multiGPU:
            pbar_iter = tqdm(total=total_iters, ascii=True, dynamic_ncols=True, leave=False)
        for i, (input_x, encoding_x, input_y, encoding_y, ground_truth) in enumerate(train_loader):
            optimizer.zero_grad()
            input_x = input_x.to(device)
            encoding_x = encoding_x.to(device)
            input_y = input_y.to(device)
            encoding_y = encoding_y.to(device)
            ground_truth = ground_truth.to(device)
            output = model(input_x, encoding_x, input_y, encoding_y)
            loss = loss_fn(output, ground_truth)
            loss.backward()
            optimizer.step()
            if (multiGPU and local_rank == 0) or not multiGPU:
                pbar_iter.set_postfix_str('loss:{:.4f}'.format(loss.item()))
                pbar_iter.update(1)
        if (multiGPU and local_rank == 0) or not multiGPU:
            pbar_iter.close()

        # validate
        if (multiGPU and local_rank == 0) or not multiGPU:
            model.eval()
            output_list = []
            gt_list = []
            pbar_iter = tqdm(total=len(valid_loader), ascii=True, dynamic_ncols=True, leave=False)
            pbar_iter.set_description_str('validating')
            with torch.no_grad():
                for i, (input_x, encoding_x, input_y, encoding_y, ground_truth) in enumerate(valid_loader):
                    input_x = input_x.to(device)
                    encoding_x = encoding_x.to(device)
                    input_y = input_y.to(device)
                    encoding_y = encoding_y.to(device)
                    output = model(input_x, encoding_x, input_y, encoding_y)
                    output_list.append(output.cpu())
                    gt_list.append(ground_truth)
                    pbar_iter.update()
            pbar_iter.close()
            if torch.__version__ > '1.13.0':
                output_list = torch.concatenate(output_list, dim=0)
                gt_list = torch.concatenate(gt_list, dim=0)
            else:
                output_list = torch.cat(output_list, dim=0)
                gt_list = torch.cat(gt_list, dim=0)
            validate_loss = loss_fn(output_list, gt_list).item()
            validate_loss_list.append(validate_loss)
            pbar_epoch.set_postfix_str('validate_loss:{:.4f}'.format(validate_loss))
            pbar_epoch.update(1)
            if validate_loss < minium_loss:
                last_save_step = epoch
                minium_loss = validate_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                pbar_epoch.set_description_str('saved at epoch %d %.4f' % (epoch + 1, minium_loss))
        if multiGPU:
            dist.barrier()
    if (multiGPU and local_rank == 0) or not multiGPU:
        pbar_epoch.close()

    # test
    if (multiGPU and local_rank == 0) or not multiGPU:
        pbar_iter = tqdm(total=len(test_loader), ascii=True, dynamic_ncols=True)
        pbar_iter.set_description_str('testing')
        output_list = []
        gt_list = []
        if best_model:
            model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
        model.eval()
        with torch.no_grad():
            for i, (input_x, encoding_x, input_y, encoding_y, ground_truth) in enumerate(test_loader):
                input_x = input_x.to(device)
                encoding_x = encoding_x.to(device)
                input_y = input_y.to(device)
                encoding_y = encoding_y.to(device)
                output = model(input_x, encoding_x, input_y, encoding_y)
                output_list.append(output.cpu())
                gt_list.append(ground_truth)
                pbar_iter.update(1)
        pbar_iter.close()
        if torch.__version__ > '1.13.0':
            output_list = torch.concatenate(output_list, dim=0)
            gt_list = torch.concatenate(gt_list, dim=0)
        else:
            output_list = torch.cat(output_list, dim=0)
            gt_list = torch.cat(gt_list, dim=0)
        test_loss = loss_fn(output_list, gt_list).item()
        mae_loss = torch.mean(torch.abs(output_list - gt_list)).item()
        print('\033[32mmse loss:{:.4f} mae loss:{:.4f}\033[0m'.format(test_loss, mae_loss))
        result_dict = arg_dict
        result_dict['mse'] = test_loss
        result_dict['mae'] = mae_loss
        print(json.dumps(result_dict, ensure_ascii=False), file=open(os.path.join(save_dir, 'result.json'), 'w'))
        if delete_model_dic:
            os.remove(os.path.join(save_dir, 'best_model.pth'))
            print('\033[33mdeleted model.pth\033[0m')
        if multiGPU:
            dist.destroy_process_group()
