# -*- coding: utf-8 -*-

# @CreateTime : 2024/2/24 024 14:48
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run.py
# @Description :
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from recbole.quick_start import run_recbole

config_dict = {
    "embedding_size": 64,  # (int)：项目的嵌入大小。默认为 64。
    "hidden_size": 128,  # (int)：处于隐藏状态的要素数。默认为 。128
    "num_layers": 1,  # (int)：GRU中的层数。默认为 。1
    "dropout_prob": 0.3,  # (float)：辍学率。默认为 。0.3
    "loss_type": "BPR",  # (str)：损失函数的类型。
    "train_neg_sample_args": {'distribution': 'uniform', 'sample_num': 1},
    # "loss_type":  "CE",  # (str)：损失函数的类型。
    # "train_neg_sample_args": None,
}
run_recbole(model="GRU4Rec", dataset="ml-100k", config_dict=config_dict)

# --END--
