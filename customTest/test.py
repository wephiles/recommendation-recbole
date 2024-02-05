# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/19 019 21:13
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/test.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

import requests
from bs4 import BeautifulSoup


def main() -> int:
    # dict_ = {
    #     "name": "computer",
    #     "age": "100",
    #     "type list": ["mobile phone", "personal computer", "desktop computer"]
    # }
    # import pickle
    # import pprint
    # with open("hello.pkl", "wb") as fp:
    #     pickle.dump(dict_, fp)

    # with open("hello.pkl", "rb") as fp:
    #     data = pickle.load(fp)
    #     pprint.pprint(data)

    # from collections import defaultdict
    # d = defaultdict(int)
    # print(d['key'])
    # d['key'] += 1
    # print(d['key'])
    # set_ = (12, 30, )
    # print(dict_.setdefault("goods", set_))
    # print(dict_)

    return 0


def get_movie_names():
    url = "https://movie.douban.com/top250"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    movie_list = soup.find_all('div', class_='hd')
    for movie in movie_list:
        movie_name = movie.find('span', class_='title').text
        print(movie_name, )


if __name__ == "__main__":
    # get_movie_names()
    main()

# END
