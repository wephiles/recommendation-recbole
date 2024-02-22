# -*- coding: utf-8 -*-
# @CreateTime : 2024/2/22 022 14:48
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run.py
# @Description : 推荐系统的深度矩阵分解模型
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from recbole.quick_start import run_recbole

run_recbole(model="DMF", dataset="ml-1m", config_file_list=[r"./example.yaml"])

# END
