# -*- coding: utf-8 -*-
# @CreateTime : 2024/3/1 001 15:30
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles


from recbole.quick_start import run_recbole

config_dict = {
    "embedding_size": 10,  # : The embedding size of features. Defaults to 10.
    "mlp_hidden_size": [256, 256, 256],  # : The hidden size of MLP layers. Defaults to [256,256,256].
    "cross_layer_num": 6,  # : The number of cross layers. Defaults to 6.
    "reg_weight": 2,  # : The L2 regularization weight. Defaults to 2.
    "dropout_prob": 0.2  # : The dropout rate. Defaults to 0.2.
}

run_recbole(model='DCN', dataset='ml-100k', config_dict=config_dict)

# --END--
