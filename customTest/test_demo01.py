# -*- coding: utf-8 -*-
# @CreateTime : 2024/2/2 002 21:39
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/test_demo01.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles


def main() -> int:
    from operator import itemgetter
    test_list = {
        "name": ("a", "b", "c", "d"),
        "age": [1, 2, 3, 4],
        "grade": (5, 6, 7, 8),
    }

    data_test = itemgetter("name", "age", "grade")
    # for item in test_list:
    #     print(data_test(item))
    print(data_test(test_list))
    print(type(data_test(test_list)))
    return 0


if __name__ == '__main__':
    main()

# END
