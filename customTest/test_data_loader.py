# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/18 018 14:44
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/test_data_loader.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

import torch
from recbole.data import Dataset, Sampler
from recbole.config import Config
from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader, DataLoaderType, Interaction


class MyDataLoader(AbstractDataLoader):
    dl_type = DataLoaderType.ORIGIN

    def __init__(self, config, dataset, sampler, shuffle=False):
        if shuffle is False:
            shuffle = True
            print('MyDataLoader must shuffle the data.')

        self.uid_field = dataset.uid_field
        self.user_list = Interaction({self.uid_field: torch.arange(dataset.user_num)})

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


# # 创建一个配置对象
# config = Config(model='BPR', dataset='ml-100k')
#
# # 创建数据集对象
# dataset = Dataset(config)
#
# # 创建采样器对象
# sampler = Sampler(config, dataset)
#
# # 创建MyDataLoader实例
# my_dataloader = MyDataLoader(config, dataset, sampler)
#
# # 迭代遍历MyDataLoader
# for batch_data in my_dataloader:
#     user_ids = batch_data[dataset.uid_field]
#     print(user_ids)

# END
