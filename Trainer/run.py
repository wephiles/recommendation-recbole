# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/24 024 15:09
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

import argparse

from recbole.quick_start import run_recbole


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='FM', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list)

# END
