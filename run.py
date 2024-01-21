# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/11 011 17:21
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from recbole.quick_start import run_recbole

# from recbole.config import Config
#
# config = Config(model='LR', dataset='ml-100k')
# run_recbole(model='Pop', dataset='ml-100k')
from recbole.quick_start import run_recbole

run_recbole(dataset='BPR', model='ml-100k', config_file_list={}, config_dict={})

# END
