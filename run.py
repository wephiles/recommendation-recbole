# -*- coding: utf-8 -*-
# @CreateTime : 2024/3/2 002 21:12
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run.py
# @Description :
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles


# from recbole.quick_start import run_recbole
# run_recbole(model="BPR", dataset="ml-100k")
from recbole.quick_start import run_recbole
# config_dict = {
#     "n_iterations": 2,
#     "n_layers": 1,
#     "reg_weight": 1e-03,
#     "cor_weight": 0.01,
#     "embedding_size": 64,
#     "n_factors": 4,
#
# }

run_recbole(model='BPR', dataset='ml-100k',)

# --END--
