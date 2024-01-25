# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/24 024 13:30
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/add_sample_test.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from logging import getLogger
from recbole.utils import init_logger, init_seed
from new_trainer1 import NewTrainer
from new_model import NewModel
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data import create_dataset
from new_sample import KGSampler

if __name__ == '__main__':
    config = Config(model=NewModel, dataset='ml-100k', config_file_list=['example.yaml'])
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # 读取item数据
    items_file = "ml-100k/u.item"
    items = {}
    with open(items_file, "r", encoding="ISO-8859-1") as f:
        for line in f:
            fields = line.strip().split("|")
            item_id = int(fields[0])
            item_name = fields[1]
            items[item_id] = item_name

    # 读取训练数据
    train_file = "ml-100k/u1.base"
    dataset = []
    with open(train_file, "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            user_id = int(fields[0])
            item_id = int(fields[1])
            dataset.append({"user_id": user_id, "item_id": item_id})

    # 创建自定义的采样器实例
    sampler = KGSampler(dataset)
    # 测试采样器
    user_ids = [1, 2, 3]  # 替换为你想要测试的用户ID列表
    num_samples = 5  # 替换为你想要采样的负样本数量

    for user_id in user_ids:
        sampled_items = sampler.sample_by_user_id(user_id, num=num_samples)
        print(f"Sampled items for user ID {user_id}:")
        for item_id in sampled_items:
            item_name = items[item_id]
            print(f"Item ID: {item_id}, Item Name: {item_name}")
        print()

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = NewModel(config, train_data.dataset).to(config['device'])
    logger.info(model)



    # trainer loading and initialization
    trainer = NewTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

# END
