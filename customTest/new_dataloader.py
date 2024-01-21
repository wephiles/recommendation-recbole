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
from torch.utils.data import DataLoader


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

    def generate_train_loader(self):
        train_dataset = self.dataset  # 直接使用 self.dataset 作为训练数据集
        # num_workers = self.config.workers if hasattr(self.config, 'workers') else 0  # 获取 workers 参数，如果不存在则设置为 0
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.training_batch_size,
            # shuffle=self.shuffle,
            shuffle=False,
            sampler=self.sampler,  # 直接将 RandomSampler 对象传递给 sampler 参数
            num_workers=self.config.num_workers
        )

        return train_dataloader

    def generate_valid_loader(self):
        valid_dataset = self.dataset  # 直接使用 self.dataset 作为验证数据集

        num_workers = self.config.workers if hasattr(self.config, 'workers') else 0  # 获取 workers 参数，如果不存在则设置为 0

        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=self.config.valid_batch_size,
            shuffle=False,  # 或根据需要设置为 True 或 False
            num_workers=self.config.num_workers
        )

        return valid_dataloader

    def generate_test_loader(self):
        test_dataset = self.dataset  # 直接使用 self.dataset 作为测试数据集

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,  # 或根据需要设置为 True 或 False
            num_workers=self.config.num_workers
        )

        return test_dataloader

# END
