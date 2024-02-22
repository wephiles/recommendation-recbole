# -*- coding: utf-8 -*-
# @CreateTime : 2024/2/18 018 13:50
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from recbole.quick_start import run_recbole
run_recbole(model='NeuMF', dataset='ml-100k', config_file_list=["config_neumf.yaml"])

# END
