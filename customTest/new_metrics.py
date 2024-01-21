# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/19 019 15:46
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/new_metrics.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles

from recbole.evaluator.base_metric import AbstractMetric
from recbole.utils import EvaluatorType


class MyMetric(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.items', 'data.num_items']
    smaller = True

    def __init__(self, config):
        ...

    def calculate_metric(self, dataobject):
        """Get the dictionary of a metric.

        Args:
            dataobject(DataStruct): it contains all the information needed to calculate metrics.

        Returns:
            dict: such as ``{'mymetric@10': 3153, 'mymetric@20': 0.3824}``
        """
        rec_items = dataobject.get('rec.items')
        # Add the logic of your metric here.

        return result_dict

# END
