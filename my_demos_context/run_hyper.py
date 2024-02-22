# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/12 012 14:37
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/run_hyper.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from logging import getLogger

from recbole.data import create_dataset, data_preparation
from recbole.trainer import HyperTuning
from recbole.config import Config

# from recbole.quick_start import objective_function
from recbole.utils import init_seed, init_logger, get_model, get_trainer


def objective_function():
    # 配置初始化
    config = Config(model="BPR", dataset="ml-100k")

    # 日志初始化
    init_logger(config)
    logger = getLogger()

    # 初始化随机种子 确保实验的可重复性
    init_seed(config['seed'], config['reproducibility'])
    # config = Config(config_dict=config_dict, config_file_list=config_file_list)
    # init_seed(config['seed'])

    # 数据集创建并筛选（过滤）
    dataset = create_dataset(config)
    logger.info(dataset)
    logger.info(dataset)

    # dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 模型加载并且初始化
    model_name = config['model']

    model = get_model(model_name)(config, train_data._dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False)
    test_result = trainer.evaluate(test_data)

    return {
        'model': model_name,
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


hp = HyperTuning(objective_function=objective_function, algo='exhaustive', early_stop=10,
                 max_evals=100, params_file='../customTest/example.yaml', fixed_config_file_list=['example.yaml'])

# 运行
hp.run()

# 把结果导出到文件中
hp.export_result(output_file='./hyper_example.result')

# 打印出最佳参数
print('best params:', hp.best_params)

# 打印出最佳结果
print('best result:')
print(hp.params2result[hp.best_params])

# END
