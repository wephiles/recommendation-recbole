# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/24 024 15:00
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/new_trainer.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from recbole.trainer import Trainer
import torch.cuda.amp as amp


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
                loss = self.model.calculate_loss1(interaction)
                self._check_nan(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        else:
            for batch_idx, interaction in enumerate(train_data):
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.calculate_loss2(interaction)
                self._check_nan(loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        return total_loss


class NewTrainer1(Trainer):
    def __init__(self, config, model):
        super(NewTrainer1, self).__init__(config, model)


def _train_epoch(self, train_data, epoch_idx, *args, **kwargs):
    self.model.train()
    scaler = amp.GradScaler(enabled=self.enable_scaler)
    for batch_idx, interaction in enumerate(iter_data):
        interaction = interaction.to(self.device)
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
            losses = loss_func(interaction)
        total_loss = losses.item() if total_loss is None else total_loss + losses.item()
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()


class NewTrainer3(Trainer):
    def __init__(self, config, model):
        super(NewTrainer3, self).__init__(config, model)
        self.optimizer = self._build_optimizer()

    def _train_epoch(self, train_data, epoch_idx, *args, **kwargs):
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
