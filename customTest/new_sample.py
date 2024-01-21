# -*- coding: utf-8 -*-
# @CreateTime : 2024/1/18 018 15:06
# @Author : 瑾瑜@20866
# @IDE : PyCharm
# @File : recommendationAlgorithms/new_sample.py
# @Description : 
# @Interpreter : python 3.10
# @Motto : You must take your place in the circle of life!
# @Site : https://github.com/wephiles or https://gitee.com/wephiles


from recbole.sampler.sampler import AbstractSampler
import numpy as np


class KGSampler(AbstractSampler):  # 定义了一个名为KGSampler的类，继承自AbstractSampler类。
    def __init__(self, dataset, distribution='uniform'):
        self.dataset = dataset  # 将传入的dataset对象保存为sampler对象的属性，以便后续使用。

        # 将dataset对象的head_entity_field和tail_entity_field属性保存为sampler对象的属性。
        self.hid_field = dataset.head_entity_field
        self.tid_field = dataset.tail_entity_field

        # 将dataset对象的head_entities和tail_entities属性保存为sampler对象的属性。
        self.hid_list = dataset.head_entities
        self.tid_list = dataset.tail_entities

        # 将dataset对象的head_entities属性转换为集合，并保存为sampler对象的head_entities属性。
        # 将dataset对象的entity_num属性保存为sampler对象的entity_num属性。
        self.head_entities = set(dataset.head_entities)
        self.entity_num = dataset.entity_num

        # 调用父类AbstractSampler的构造函数，并传递distribution参数。
        super().__init__(distribution=distribution)

    def _uni_sampling(self, sample_num):
        # 定义了一个名为_uni_sampling的私有方法，用于在均匀分布下进行采样。它接受一个sample_num参数，表示要采样的数量。

        return np.random.randint(1, self.entity_num, sample_num)
        # 使用np.random.randint函数在1到self.entity_num之间生成sample_num个随机整数，并返回结果。

    def _get_candidates_list(self):
        # 定义了一个名为_get_candidates_list的私有方法，用于获取候选实体列表。
        return list(self.hid_list) + list(self.tid_list)  # 将hid_list和tid_list合并为一个列表，并返回结果。

    def get_used_ids(self):  # 获取已使用的实体ID。

        # 创建一个包含self.entity_num个空集合的NumPy数组，并将其赋值给used_tail_entity_id变量。
        used_tail_entity_id = np.array([set() for _ in range(self.entity_num)])

        # 遍历hid_list和tid_list，将每个tid添加到对应的hid索引处的集合中。
        for hid, tid in zip(self.hid_list, self.tid_list):
            used_tail_entity_id[hid].add(tid)
        # 检查每个集合是否与所有实体相关联。如果某个集合的大小加1等于self.entity_num，表示某个头实体与所有实体存在关系，
        # 不能为其采样负实体。在这种情况下，抛出ValueError异常。
        for used_tail_set in used_tail_entity_id:
            if len(used_tail_set) + 1 == self.entity_num:  # [pad] is a entity.
                raise ValueError(
                    'Some head entities have relation with all entities, '
                    'which we can not sample negative entities for them.'
                )

        # 返回used_tail_entity_id变量，其中包含了每个头实体对应的已使用的尾实体集合。
        return used_tail_entity_id

    def sample_by_entity_ids(self, head_entity_ids, num=1):
        # 定义了一个名为sample_by_entity_ids的公共方法，用于根据给定的头实体ID列表进行采样。
        # 它接受一个head_entity_ids参数表示头实体的ID列表，以及一个可选的num参数表示采样的数量，默认为1。
        try:
            # 尝试调用sampler对象的sample_by_key_ids方法进行采样，其中传入head_entity_ids和num作为参数。
            # 如果发生IndexError异常，说明head_entity_ids中的某个头实体ID不存在，此时抛出ValueError异常并提供相应的错误消息。
            return self.sample_by_key_ids(head_entity_ids, num)
        except IndexError:
            for head_entity_id in head_entity_ids:
                if head_entity_id not in self.head_entities:
                    raise ValueError(f'head_entity_id [{head_entity_id}] not exist.')


# class Dataset:
#     def __init__(self, head_entity_field, tail_entity_field, head_entities, tail_entities, entity_num):
#         self.head_entity_field = head_entity_field
#         self.tail_entity_field = tail_entity_field
#         self.head_entities = head_entities
#         self.tail_entities = tail_entities
#         self.entity_num = entity_num
#
#
# # 创建虚拟数据集对象
# dataset = Dataset("hid_field", "tid_field", [1, 2, 3], [4, 5, 6], 7)
#
# # 创建KGSampler实例
# sampler = KGSampler(dataset, distribution='uniform')
#
# # 例如，使用sample_by_entity_ids方法测试
# head_entity_ids = [1, 2]
# num_samples = 2
#
# samples = sampler.sample_by_entity_ids(head_entity_ids, num=num_samples)
# print(samples)

# END
