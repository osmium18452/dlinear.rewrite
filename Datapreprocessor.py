import numpy as np
import torch
from torch.utils.data import Dataset


class Datapreprocessor:
    def __init__(self, dataset, input_len, output_len, stride=1, train_set_ratio=.6, validate_set_ratio=.2):
        self.num_sensors = dataset.shape[1]
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = self.input_len + self.output_len
        self.std = np.std(dataset, axis=0).reshape([1, -1])
        self.mean = np.mean(dataset, axis=0).reshape([1, -1])
        self.dataset = (dataset - self.mean) / self.std

        self.num_samples = len(np.arange(0, dataset.shape[0] - self.total_len + 1, stride))
        sample_indices = np.arange(0, dataset.shape[0] - self.total_len + 1, stride)
        sample_window = np.arange(0, self.total_len)
        sample_index_mask = np.repeat(sample_indices, self.total_len).reshape(self.num_samples, -1) + sample_window
        samples = self.dataset[sample_index_mask]
        input = samples[:, :self.input_len, :]
        ground_truth = samples[:, self.input_len:, :]
        self.num_train_samples = int(self.num_samples * train_set_ratio)
        self.num_validate_samples = int(self.num_samples * validate_set_ratio)
        self.num_test_samples = self.num_samples - self.num_train_samples - self.num_validate_samples

        self.train_input = input[:self.num_train_samples]
        self.train_ground_truth = ground_truth[:self.num_train_samples]
        self.validate_input = input[self.num_train_samples:self.num_train_samples + self.num_validate_samples]
        self.validate_ground_truth = ground_truth[
                                     self.num_train_samples:self.num_train_samples + self.num_validate_samples]
        self.test_input = input[self.num_train_samples + self.num_validate_samples:]
        self.test_ground_truth = ground_truth[self.num_train_samples + self.num_validate_samples:]

    def load_train_samples(self):
        return torch.Tensor(self.train_input), torch.Tensor(self.train_ground_truth)

    def load_validate_samples(self):
        return torch.Tensor(self.validate_input), torch.Tensor(self.validate_ground_truth)

    def load_test_samples(self):
        return torch.Tensor(self.test_input), torch.Tensor(self.test_ground_truth)


class TSDataset(Dataset):
    def __init__(self, input, ground_truth):
        super(TSDataset, self).__init__()
        self.input = input
        self.ground_truth = ground_truth

    def __getitem__(self,index):
        return self.input[index], self.ground_truth[index]

    def __len__(self):
        return self.input.shape[0]

