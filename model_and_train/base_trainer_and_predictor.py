import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mapper.train_and_predictor_mapper import select_rating_by_user_id, select_user_having_ratings_greater_equal
from model_and_train.id_mapping import IdMapping


class BaseDeepLearningSystem:
    def __init__(self, train_ratio, lr, batch_size, epochs, params_saved_path):
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params_saved_path = params_saved_path
        print('当前设备:', self.device)

        # 超参数
        self.train_ratio = train_ratio
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs


    '''训练模型'''
    def train(self, model=None, dataloader=None):
        pass


    '''测试模型'''
    def test(self, model=None, dataloader=None):
        pass


    '''增量训练'''
    def update_train(self, newly_update_time: str):
        pass


    '''预测'''
    def predict(self, user_ids: list, book_ids: list):
        pass


    '''召回'''
    def recall(self, user_id: int, k: int = 10, threshold: float = 4.5):
        pass

    '''top_k测试 基于预测结果的(全负)召回率与准确率测试'''
    def test_top_k(self, min_rating_num: int = 200, k: int = 100, threshold: float = 4.5) -> tuple[
        float, float, float]:
        """
        :param min_rating_num: 最小发表的评价数量
        :param k: 选取k个作为推荐(召回)结果
        :param threshold: 最低评分阈值
        :return: avg_precision 精确度, avg_precision_full 全负精确度, avg_recall 召回率
        """
        print("基于预测结果的top-K测试开始")
        user_ids = select_user_having_ratings_greater_equal(min_rating_num)
        avg_precision = 0
        avg_precision_full = 0
        avg_recall = 0
        no_tpfp_user_num = 0

        for user_id in tqdm(user_ids):
            recall_list = self.recall(user_id, k, threshold)
            pred = {book_id for book_id, rating in recall_list}  # pred是预测结果
            pos = set()
            neg = set()

            # 找到该 user 有多少个真实数据 rating >= threshold，将结果作为 set 记录下来，记 set 为 pos, 否则放入 neg集合
            book_ratings = select_rating_by_user_id(user_id)  # 获取用户评分数据 函数返回 [(book_id, rating), ...] 的列表
            for book_rating in book_ratings:
                book_id, rating = book_rating
                pos.add(book_id) if rating >= threshold else neg.add(book_id)

            TP = len(pred & pos)  # TP: 推荐列表中用户喜欢的书籍的数量
            FP = len(pred & neg)  # FP: 推荐列表中用户不喜欢的书籍的数量
            P = len(pos)  # P: 用户喜欢的全部书籍的数量

            if (TP + FP) > 0:
                precision = TP / (TP + FP)
                avg_precision += precision
            else:
                no_tpfp_user_num += 1  # Precision 准确率 = TP / (TP + FP)
            precision_full = TP / len(pred) if len(pred) > 0 else 0  # Precision_Full 全负准确率 = TP / (TP + FP)
            recall = TP / P if P > 0 else 0  # Recall 召回率 = TP / P
            avg_precision_full += precision_full
            avg_recall += recall

        avg_precision /= (len(user_ids) - no_tpfp_user_num)
        avg_precision_full /= len(user_ids)
        avg_recall /= len(user_ids)
        print(f"预测列表上的top-K精确率: {avg_precision}")
        print(f"预测列表上的top-K全负准确率: {avg_precision_full}")
        print(f"预测列表上的top-K召回率: {avg_recall}")
        print("基于预测结果的top-K测试结束!")
        return avg_precision, avg_precision_full, avg_recall

    '''获取数据集'''
    def __get_dataset__(self, newly_update_time: str):
        """
        子类可以只实现这个而不用重写__get_xxx_dataloader__
        """
        pass


    '''加载模型参数'''
    def __load_model_parameters__(self, model):
        pass


    def __save_model_parameters__(self, model, epoch):
        """
        保存模型参数
        """
        state = {'model': model.state_dict(), 'epoch': epoch}
        torch.save(state, self.params_saved_path)
        print(f"模型已保存, epoch={epoch + 1}")


    def __get_train_and_test_dataloader__(self) -> tuple[DataLoader, DataLoader]:
        """
        获取train + test dataloader
        :return: train_dataloader, test_dataloader
        """
        dataset = self.__get_dataset__('1970-01-01')

        if len(dataset) == 0:
            print('暂无训练数据')
            return

        total_size = len(dataset)
        train_size = int(self.train_ratio * total_size)
        test_size = total_size - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],
                                                                    torch.manual_seed(123))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=2,
                                      shuffle=True, drop_last=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=2,
                                     shuffle=True, drop_last=False)
        print('已加载并分割数据集')
        return train_dataloader, test_dataloader


    def __get_full_dataloader__(self, newly_update_time) -> DataLoader:
        """
        获取完整的数据集
        :param: newly_update_time 加载在该时间后的所有数据
        :return: dataloader
        """
        dataset = self.__get_dataset__(newly_update_time)

        if len(dataset) == 0:
            print('暂无训练数据')
            return
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=2,
                                shuffle=True, drop_last=False)
        print('已加载数据集')
        return dataloader


    def  __get_id_mapping__(self) -> tuple[IdMapping, int, int, int]:
        """
        加载id映射关系
        :return: new_id_mapping, n_users, n_books, n_tags
        """
        new_id_mapping = IdMapping()  # 创建新的mapping
        n_users, n_books, n_tags = new_id_mapping.n_users, new_id_mapping.n_books, new_id_mapping.n_tags
        print('已加载id映射关系')
        return new_id_mapping, n_users, n_books, n_tags
