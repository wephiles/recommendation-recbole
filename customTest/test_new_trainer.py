# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/17 017 16:47
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/test_new_trainer.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.utils import init_seed, init_logger
from new_trainer1 import NewTrainer


def train(config_dict):
    config = Config(model=BPR, dataset='ml-100k', config_dict=config_dict, config_file_list=['example.yaml'])

    # 初始化
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    dataset = create_dataset(config)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 创建模型实例
    model = BPR(config, train_data.dataset).to(config['device'])

    # 初始化你的训练器
    trainer = NewTrainer(config, model)

    # 训练模型
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    print("best_valid_score: ", best_valid_score)
    print("best_valid_result: ", best_valid_result)


def test_trainer1(config_dict):
    train(config_dict)


def test_trainer2(config_dict):
    train(config_dict)


def test_trainer3(config_dict):
    train(config_dict)


if __name__ == '__main__':
    # config_dict1 = {
    #     'model': 'BPR',  # 这里是示例，根据需要选择模型
    #     'dataset': 'ml-100k',  # 选择或指定数据集
    #     'trainer': 'NewTrainer',  # 使用你的自定义训练器
    #     # 其他配置参数...
    # }
    # test_trainer1(config_dict1)

    # config_dict2 = {
    #     'model': 'BPR',  # 这里是示例，根据需要选择模型
    #     'dataset': 'ml-100k',  # 选择或指定数据集
    #     'trainer': 'NewTrainerMix',  # 使用你的自定义训练器
    #     # 其他配置参数...
    # }
    # test_trainer2(config_dict2)

    config_dict3 = {
        'model': 'BPR',  # 这里是示例，根据需要选择模型
        'dataset': 'ml-100k',  # 选择或指定数据集
        'trainer': 'NewTrainerLayer',  # 使用你的自定义训练器
        # 其他配置参数...
    }
    test_trainer3(config_dict3)

# END
