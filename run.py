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


main()

# END
