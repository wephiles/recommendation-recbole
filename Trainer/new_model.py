# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/13 013 13:17
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/new_model.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from recbole.model.abstract_recommender import GeneralRecommender, AbstractRecommender
import torch
import torch.nn as nn
from recbole.model.loss import BPRLoss
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class NewModel(AbstractRecommender):
    # 重新定义__init__方法,以初始化模型,包括加载数据集信息,模型参数,定义模型结构和初始化方法
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NewModel, self).__init__(config, dataset)

        # 加载数据及信息
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        # 加载参数信息
        self.embedding_size = config['embedding_size']

        # 定义层和损失
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)  # 将self.n_items写成了self.n_users
        self.loss = BPRLoss()

        # 参数初始化
        self.apply(xavier_normal_initialization)  # 将xavier_normal_initialization写成了xavier_normal_initialization()

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]  # 写成了（self.USER_ID）
        pos_item = interaction[self.ITEM_ID]  # 写成了（self.ITEM_ID]）
        neg_item = interaction[self.NEG_ITEM_ID]  # 写成了（self.NEG_ITEM_ID）

        user_e = self.user_embedding(user)  # [batch_size, embedding_size]
        pos_item_e = self.item_embedding(pos_item)  # [batch_size, embedding_size]
        neg_item_e = self.item_embedding(neg_item)  # [batch_size, embedding_size]
        pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1)  # [batch_size]
        neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1)  # [batch_size]

        loss = self.loss(pos_item_score, neg_item_score)  # []

        return loss

    def predict(self, interaction):
        """
        is used to compute the score for a give user-item pair.
        The input is a Interaction, and the output is a score.
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e = self.user_embedding(user)  # [batch_size, embedding_size]
        item_e = self.item_embedding(item)  # [batch_size, embedding_size]

        scores = torch.mul(user_e, item_e).sum(dim=1)

        return scores

    def full_sort_predict(self, interaction):
        """
        evaluate the full ranking in the NewModel,

        :param interaction:
        :return: I do not know ! Do not ask me, SB.
        """
        user = interaction[self.USER_ID]

        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight

        scores = torch.matmul(user_e, all_item_e.transpose(0, 1))

        return scores

# END
