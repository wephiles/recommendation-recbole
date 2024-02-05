# -*- coding: utf-8 -*-
# @CreateTime : 2024/2/4 004 13:27
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run_recbole.py
# @Description : 训练模型
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles


from recbole.quick_start import run_recbole

parameter_dict = {
   'train_neg_sample_args': None,
}

run_recbole(model="GRU4Rec", dataset="ml-100k", config_file_list=["test.yaml"], config_dict=parameter_dict)

# END
