# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/19 019 15:14
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run_recbole.py
# @Description : run.py文件，用于结合自定义模型、自定义DataLoader、 自定义训练器、自定义采样器、自定义指标
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles


from recbole.quick_start import run_recbole


def run():
    run_recbole(config_file_list=['example.yaml'])


if __name__ == '__main__':
    run()

# END
