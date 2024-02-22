# -*- coding: utf-8 -*-
# @CreateTime : 2024/2/7 007 13:43
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run_recbole.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from recbole.quick_start import run_recbole

config_dict = {
    "k": 100,
    "shrink": 0.5
}

run_recbole(model="ItemKNN", dataset="ml-100k", config_dict=config_dict)

# END