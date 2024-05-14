import random

import pandas as pd
from torch.utils.data import Dataset

from mapper.dataset_mapper import select_new_tuple_user_work_rating, select_old_tuple_user_work_rating
from model_and_train.id_mapping import IdMapping

'''Dataset'''
class RatingDataset(Dataset):
    def __init__(self, newly_update_time: str = '0000-00-00', transform=None):
        self.transform = transform
        data = load_data(newly_update_time)
        self.data = IdMapping().mapping_data(data)

    def __getitem__(self, index):
        username, book_id, rating = self.data.loc[index]
        return username, book_id, rating

    def __len__(self):
        return len(self.data)



'''从数据库读取数据'''
def load_data(newly_update_time: str = '0000-00-00', mix_old_data_ratio: float = 3.0) -> pd.DataFrame:
    """
    从数据库读数据
    :param newly_update_time: 从这个时间以后的新数据 格式为 yyyy-MM-dd 或 yyyy-MM-dd HH:mm:ss
    :param mix_old_data_ratio: 以1:mix_old_data_ratio的比例混入旧数据
    :return:
    """
    results = select_new_tuple_user_work_rating(newly_update_time)      # 查询新的(user_id, work_id, rating)三元组
    # 混合老的三元组数据(1:mix_old_data_ratio比例)
    if mix_old_data_ratio > 0.0 and len(results) > 0:
        old_results = select_old_tuple_user_work_rating(newly_update_time)
        results = results + random.sample(old_results, min(len(old_results), int(mix_old_data_ratio * len(results))))
    results = pd.DataFrame(results, columns=['user_id', 'book_id', 'rating'])
    return results
