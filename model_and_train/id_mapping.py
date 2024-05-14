import random

import pandas as pd

from mapper.dataset_mapper import select_work_with_tag
from mapper.id_mapping_mapper import select_seq_user_id, select_seq_book_id, select_seq_book_tag_id, \
    select_tag_id_by_book_id, select_count_tags


class IdMapping:
    def __init__(self) -> None:
        self.user_id_map = None
        self.book_id_map = None
        self.tag_id_map = None
        self.n_users = 0
        self.n_books = 0
        self.n_tags = 0
        self.update_user_id_map()
        self.update_book_id_map()
        self.update_tag_id_map()
        self.work_tag_dict = self.update_work_tag_mapped_dict()

    def update_user_id_map(self):
        self.user_id_map = {user_id: seq_id for user_id, seq_id in select_seq_user_id()}
        self.n_users = len(self.user_id_map)

    def update_book_id_map(self):
        self.book_id_map = {book_id: seq_id for book_id, seq_id in select_seq_book_id()}
        self.n_books = len(self.book_id_map)

    def update_tag_id_map(self):
        self.tag_id_map = {tag_id: seq_id for tag_id, seq_id in select_seq_book_tag_id()}
        self.n_tags = select_count_tags()[0]

    def mapping_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # 替换pd.DataFrame中的user_id与book_id
        if data is not None:
            if 'user_id' in data:
                data['user_id'] = data['user_id'].map(self.user_id_map)
            if 'book_id' in data:
                data['book_id'] = data['book_id'].map(self.book_id_map)
        return data

    def mapping_ids(self, user_ids: list, book_ids: list) -> tuple[list, list]:
        # 替换原始数据中的user_id与book_id
        mapped_users = [self.user_id_map[user_id] for user_id in user_ids]
        mapped_works = [self.book_id_map[book_id] for book_id in book_ids]
        return mapped_users, mapped_works

    def mapping_tags(self, tag_ids: list) -> list:
        # 替换原始数据中的tag_id
        mapped_tags = [self.tag_id_map[tag_id] for tag_id in tag_ids]
        return mapped_tags

    def get_mapped_tags_by_book_id(self, book_id: int) -> list:
        # 按work_id获取的替换的tag_id
        original_tags = select_tag_id_by_book_id(book_id)
        mapped_tags = []
        for tag_id in original_tags:
            mapped_tags.append(self.tag_id_map[tag_id])
        return mapped_tags

    # 加载{work_id: tag_id}字典，字典中work_id,tag_id都已经完成映射
    def update_work_tag_mapped_dict(self) -> dict:
        # 先从数据库查出work与其对应的tag 并加载成字典
        work_tag_list = select_work_with_tag()
        work_tag_dict = dict()
        for work_tag_pair in work_tag_list:
            work_id, tag_id = work_tag_pair
            work_id = self.book_id_map[work_id]  # 将work_id进行映射

            if work_id not in work_tag_dict:    # 加入字典
                work_tag_dict[work_id] = [tag_id]
            else:
                work_tag_dict[work_id].append(tag_id)

        for work_id in work_tag_dict.keys():
            work_tag_dict[work_id] = self.mapping_tags(work_tag_dict[work_id])  # 进行tag_id映射
            if len(work_tag_dict[work_id]) > 8:  # 超过8个tag则截断(一般不发生)
                work_tag_dict[work_id] = work_tag_dict[work_id][:8]

            while len(work_tag_dict[work_id]) < 8:
                element_to_fill = random.choice(work_tag_dict[work_id])   # 随机选择一个元素用于填充
                work_tag_dict[work_id].append(element_to_fill)

        return work_tag_dict


if __name__ == '__main__':
    id_mapping = IdMapping()
    print(id_mapping.get_mapped_tags_by_book_id(book_id=1000019))
