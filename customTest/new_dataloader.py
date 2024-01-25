# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/18 018 14:39
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/new_dataloader.py
# @Description :
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

import torch
from recbole.data import AbstractDataLoader, Interaction
from logging import getLogger

from torch.utils.data import TensorDataset, DataLoader


class MyDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        if shuffle is False:
            shuffle = True
            self.logger.warning('UserDataLoader must shuffle the data.')

        self.uid_field = dataset.uid_field
        self.user_list = Interaction({self.uid_field: torch.arange(dataset.user_num)})
        self.sample_size = 100
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config['train_batch_size']
        self.step = batch_size
        self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        return len(self.user_list)

    def _shuffle(self):
        self.user_list.shuffle()

    def _next_batch_data(self):
        cur_data = self.user_list[self.pr:self.pr + self.step]
        self.pr += self.step
        return cur_data

    def generate_train_loader(self, batch_size):
        # 将数据转换为张量
        data_tensor = torch.tensor(self.data)

        # 创建数据集对象
        dataset = TensorDataset(data_tensor)

        # 创建数据加载器对象
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return train_loader

# END
