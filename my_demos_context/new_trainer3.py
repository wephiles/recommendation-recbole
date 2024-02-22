# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/18 018 14:18
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/new_trainer3.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from recbole.trainer import Trainer


class NewTrainerLayer(Trainer):
    def __init__(self, config, model):
        super(NewTrainerLayer, self).__init__(config, model)
        self.optimizer = self._build_optimizer()


def _train_epoch(self, train_data, epoch_idx):
    interaction = None
    self.model.train()
    total_loss = 0.
    for batch_idx, interaction in enumerate(train_data):
        interaction = interaction.to(self.device)
    self.optimizer.zero_grad()
    loss = self.model.calculate_loss1(interaction)
    self._check_nan(loss)
    loss.backward()
    self.optimizer.step()
    total_loss += loss.item()
    return total_loss

# END
