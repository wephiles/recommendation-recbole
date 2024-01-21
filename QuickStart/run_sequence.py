# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/20 020 15:40
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run_sequence.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from recbole.quick_start import run_recbole
run_recbole(model='GRU4Rec', dataset='ml-100k', config_file_list=['sequence_test.yaml'])

# END
