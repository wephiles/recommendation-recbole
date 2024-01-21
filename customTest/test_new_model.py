# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/13 013 13:56
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/test_new_model.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

"""
测试，
测试新模型 new_model.py 里的 NewModel
"""

from logging import getLogger
from recbole.utils import init_logger, init_seed
# from recbole.trainer import Trainer
from new_trainer1 import Trainer
from new_model import NewModel
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from new_dataloader import MyDataLoader

if __name__ == '__main__':
    config = Config(model=NewModel, dataset='ml-100k', config_file_list=['example.yaml'])
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting

    dataloader = MyDataLoader(config, dataset, sampler=None, shuffle=True)
    train_loader = dataloader.generate_train_loader()
    valid_loader = dataloader.generate_valid_loader()
    test_loader = dataloader.generate_test_loader()

    # train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    # model = NewModel(config, train_data.dataset).to(config['device'])
    model = NewModel(config, train_loader.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    # best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
    best_valid_score, best_valid_result = trainer.fit(train_loader, valid_loader)

    # model evaluation
    # test_result = trainer.evaluate(test_loader)
    test_result = trainer.evaluate(test_loader)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

# END
