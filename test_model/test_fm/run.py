# -*- coding: utf-8 -*-
# @CreateTime : 2024/2/23 023 14:36
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from recbole.quick_start import run_recbole
run_recbole(dataset="ml-1m", model="FM")

# END
