# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/12 012 14:08
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run_01.py
# @Description : 在RecBole中调用不同的模块以满足要求。
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

if __name__ == '__main__':
    # todo: 这个程序有问题，需要修改
    """
    完整流程：
    配置初始化
    初始化随机种子
    数据集筛选
    数据集拆分
    模型初始化
    训练器初始化
    自动选择模型和训练器
    模型训练 
    模型评估
    从断点恢复模型
    
    补充：从以前的模型参数训练模型
    ...

    if __name__ == '__main__':
        ...
        # trainer loading and initialization
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
        # resume from break point
        checkpoint_file = 'checkpoint.pth'
        trainer.resume_checkpoint(checkpoint_file)
    
        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
        ...
    """

    # 配置初始化
    config = Config(model="BPR", dataset="ml-100k")

    # 初始化随机种子 确保实验的可重复性
    init_seed(config['seed'], config['reproducibility'])

    # 日志初始化
    init_logger(config)
    logger = getLogger()

    # 将配置写进日志
    logger.info(config)

    # 数据集创建并筛选（过滤）
    dataset = create_dataset(config)
    logger.info(dataset)

    # 数据集拆封
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 模型加载并且初始化
    model = BPR(config, train_data.dataset).to(config['device'])
    logger.info(dataset)

    # 训练器加载和初始化
    trainer = Trainer(config, model)

    # 模型训练
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # 模型评语
    test_result = trainer.evaluate(test_data)
    print(test_result)

# END
