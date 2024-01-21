# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/17 017 16:41
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/new_trainer2.py
# @Description :
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

# import torch
# import torch.cuda.amp as amp
# from recbole.trainer import Trainer
#
#
# class NewTrainerMix(Trainer):
#
#     def __init__(self, config, model):
#         super(NewTrainerMix).__init__(config, model)
#
#     def _train_epoch(self, train_data, epoch_index, *args, **kwargs):
#         self.model.train()
#         scaler = amp.GradScaler(enabled=self.enable_scaler)
#         for batch_idx, interaction in enumerate(iter_data):
#             interaction = interaction.to(self.device)
#             self.optimizer.zero_grad()
#             with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
#                 losses = loss_func(interaction)
#             total_loss = losses.item() if total_loss is None else total_loss + losses.item()
#             scaler.scale(loss).backward()
#             scaler.step(self.optimizer)
#             scaler.update()

from recbole.trainer import Trainer
import torch.cuda.amp as amp
import torch


class NewTrainerMix(Trainer):
    def __init__(self, config, model):
        super(NewTrainerMix, self).__init__(config, model)


def _train_epoch(self, train_data, epoch_idx):
    total_loss = 0

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

# END
