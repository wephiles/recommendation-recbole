# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/17 017 14:34
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/new_trainer1.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles


from recbole.trainer import Trainer


class NewTrainer(Trainer):

    def __init__(self, config, model):
        super(NewTrainer, self).__init__(config, model)

    def _train_epoch(self, train_data, epoch_idx, *args, **kwargs):
        self.model.train()
        total_loss = 0.

        if epoch_idx % 2 == 0:
            for batch_idx, interaction in enumerate(train_data):
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.calculate_loss(interaction)
                self._check_nan(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        else:
            for batch_idx, interaction in enumerate(train_data):
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.calculate_loss(interaction)
                self._check_nan(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss

# END
