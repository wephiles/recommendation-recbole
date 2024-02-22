# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/11 011 17:21
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run_recbole.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

import pprint
import numpy as np


def main() -> int:
    """主函数

    :return:
    """
    from operator import itemgetter

    data = [
        ('John', 30),
        ('Jane', 20),
        ('Dave', 40)
    ]

    # Get name and age
    get_name_and_age = itemgetter(0, 1)
    for d in data:
        # print(get_name_and_age(d)[0])
        # print(get_name_and_age(d)[1])
        pprint.pprint(get_name_and_age(d))
    return 0


def test_scipy_demo1():
    from scipy.sparse import csr_matrix

    # 创建一个稀疏矩阵
    row = [0, 1, 2, 2]
    col = [0, 1, 2, 3]
    data = [1, 2, 3, 4]
    matrix = csr_matrix((data, (row, col)), shape=(3, 4))

    # 打印稀疏矩阵
    print(matrix.toarray())

    # 矩阵乘法
    result = matrix.dot(matrix.T)
    print(result.toarray())


# main()
# test_scipy_demo1()


def test():
    a_list = [1, 2, 3]
    a_array = np.array(a_list)
    print(-a_array)


test()

# END
