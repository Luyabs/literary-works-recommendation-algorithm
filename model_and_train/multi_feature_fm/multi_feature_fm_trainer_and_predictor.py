import os
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from mapper.train_and_predictor_mapper import select_tag_text_by_seq_work_id, select_introduction_by_seq_work_id
from model_and_train.base_trainer_and_predictor import BaseDeepLearningSystem
from model_and_train.dataset.rating_dataset import RatingDataset
from model_and_train.multi_feature_fm.mult_feature_fm import MultiFeatureFM



class MultiFeatureFMSystem(BaseDeepLearningSystem):
    def __init__(self, train_ratio=0.9, dim=80, lr=0.0001, batch_size=256, epochs=5,
                 params_saved_path=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
                                                'model_params', 'deep_fm_model_adamW_mae_dim=80_epoch=20.pth')):
        print('### 正在初始化MultiFeatureFM算法 ###')
        super().__init__(train_ratio, lr, batch_size, epochs, params_saved_path)

        # user_id -> seq_id / book_id -> seq_id 映射关系加载
        self.id_mapping, n_users, _, n_tags = self.__get_id_mapping__()

        # 预训练 BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('base_bert_chinese')

        # 数据集加载 (这些数据集不用于update_train, 仅用于初始train与test)
        self.train_dataloader, self.test_dataloader = self.__get_train_and_test_dataloader__()

        # 模型 + 加载参数
        self.dim = dim
        self.model = MultiFeatureFM(n_users, n_tags, dim).to(self.device)
        self.__load_model_parameters__(self.model)

        # 损失函数和优化器
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def preprocess_text(self, texts, max_length=80):
        """ 对文本进行 tokenization, padding, truncation """
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )


    '''通过 book_id 加载 book_tag_ids'''
    def __get_tag_ids_by_book_id__(self, book_ids: torch.Tensor) -> torch.Tensor:
        """
        通过 book_id 加载 book_tags
        :param book_id: 书本id 张量 shape = [batch_size]
        :return: tag_ids 张量 shape = [batch_size, 8]
        """
        tag_ids = []
        for book_id in book_ids:
            if book_id.item() not in self.id_mapping.work_tag_dict:
                print(book_id.item())
                continue
            tag_ids.append(self.id_mapping.work_tag_dict[book_id.item()])  # book_ids 是tensor [batch_size]
        tag_ids = torch.LongTensor(tag_ids).to(self.device)  # list -> tensor
        return tag_ids

    '''通过 book_id 加载 book_tags'''
    def __get_tags_by_book_id__(self, book_ids: torch.Tensor) -> list[Any]:
        """
        通过 book_id 加载 book_tags
        :param book_id: 书本id 张量 shape = [batch_size]
        :return: tags 张量 shape = [batch_size, 8, 768]
        """
        tags = []
        for book_id in book_ids:
            if book_id.item() not in self.id_mapping.work_tag_dict:
                print(book_id.item())
                continue
            tag = select_tag_text_by_seq_work_id(book_id.item())
            tags.append(tag[0] if tag is not None and tag != '' else 'no tag')
        # tags = torch.tensor(tags).to(self.device)
        return tags

    '''通过 book_id 加载 book_introduction'''
    def __get_introduction_by_book_id__(self, book_ids: torch.Tensor) -> list[Any]:
        """
        通过 book_id 加载 book_introduction
        """
        introductions = []
        for book_id in book_ids:
            if book_id.item() not in self.id_mapping.work_tag_dict:
                print(book_id.item())
                continue
            introduction = select_introduction_by_seq_work_id(book_id.item())
            introductions.append(introduction[0] if introduction is not None and introduction != '' else 'no introduction')
        # introductions = torch.tensor(introductions).to(self.device)
        return introductions

    '''训练模型'''
    def train(self, model=None, dataloader=None) -> None:
        """
        训练模型
        :param model: 如果不指定将使用本类的成员变量 self.model
        :param dataloader: 如果不指定将使用本类的成员变量 self.train_dataloader
        :return: 无
        """
        print('DeepFM 开始训练')

        if model is None:
            model = self.model
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        if dataloader is None:
            self.train_dataloader, self.test_dataloader = self.__get_train_and_test_dataloader__()
            dataloader = self.train_dataloader

        for epoch in range(self.epochs):
            for batch_idx, (user, book, rating) in enumerate(dataloader):
                # 处理三元组信息
                user = user.long().to(self.device)
                tag_ids = self.__get_tag_ids_by_book_id__(book).to(self.device)
                click = torch.where(rating >= 4, 1.0, 0.0).to(self.device)  # 评分 ≥ 4 认为会点击

                # 获取书本描述 & 标签文本
                raw_introduction = self.__get_introduction_by_book_id__(book)
                raw_tags = self.__get_tags_by_book_id__(book)

                # 对书本描述进行 BERT 预处理
                introduction_tokens = self.preprocess_text(raw_introduction)
                introduction_input = introduction_tokens["input_ids"].to(self.device)
                introduction_mask = introduction_tokens["attention_mask"].to(self.device)

                # 对标签文本进行 BERT 预处理
                tags_tokens = self.preprocess_text(raw_tags)
                tags_input = tags_tokens["input_ids"].to(self.device)
                tags_mask = tags_tokens["attention_mask"].to(self.device)
                click_hat, contrastive_loss = model(user, tag_ids, introduction_input, introduction_mask , tags_input, tags_mask)  # 前向传播
                pred_loss = self.criterion(click_hat, click) # 计算损失
                loss = 0.5 * pred_loss + 0.5 * contrastive_loss

                # 反向传播与优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{self.epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {round(loss.item(), 4)}')
            self.__save_model_parameters__(model, epoch)
        print("DeepFM 训练完成!")

    # '''测试模型'''
    # def test(self, model=None, dataloader=None) -> tuple[float, float, float, float, float]:
    #     """
    #     测试模型
    #     :param model: 如果不指定将使用本类的成员变量 self.model
    #     :param dataloader: 如果不指定将使用本类的成员变量 self.test_dataloader
    #     """
    #     print('DeepFM 开始测试')
    #     model = self.model if model is None else model
    #     dataloader = self.test_dataloader if dataloader is None else dataloader
    #
    #     self.model.eval()
    #     total_size = len(dataloader) * dataloader.batch_size
    #     bce_loss = 0
    #     mae_loss = 0
    #     accuracy = 0
    #
    #     true_pos = 0
    #     prediction = 0
    #     pos = 0
    #
    #     with torch.no_grad():
    #         for batch_idx, (user, book, rating) in enumerate(dataloader):
    #             user = user.long().to(self.device)
    #             tags = self.__get_tags_by_book_id__(book).to(self.device)
    #             click = torch.where(rating >= 4, 1.0, 0.0).to(self.device)  # 评分 ≥ 4 认为会点击 即 accuracy = 0.1
    #
    #             click_hat = model(user, tags)  # 前向传播
    #             loss = self.criterion(click_hat, click)  # 计算损失
    #
    #             # 预估点击率 ≥ 0.5 视为会击, accuracy 统计rating_hat与rating是否一致
    #             accuracy += ((click_hat >= 0.5) == (click == 1.0)).sum().item()
    #             bce_loss += loss.item()
    #             mae_loss += abs(click_hat - click).sum().item()
    #
    #             # 计算混淆矩阵元素
    #             true_pos += ((click_hat >= 0.5) & (click == 1.0)).sum().item()  # TP 该batch中预测正确的正例
    #
    #             prediction += (click_hat >= 0.5).sum().item()  # 模型的预测结果Prediction  该batch中预测的正例 Prediciton = TP + FP
    #             pos += (click == 1.0).sum().item()  # P 该batch全部的正例 P = TP + FN
    #
    #     bce_loss /= len(dataloader)  # MSE Loss 用于训练时损失函数
    #     mae_loss /= total_size  # MAE Loss 用于直观反馈数据预测偏差
    #     accuracy /= total_size
    #     precision = true_pos / prediction if prediction > 0 else 0  # Precision 准确率 = TP / Prediction = TP / (TP + FP)
    #     recall = true_pos / pos if pos > 0 else 0  # Recall 召回率 = TP / P = TP / (TP + FN)
    #     print(f"测试集上的平均BCE Loss: {bce_loss}")
    #     print(f"测试集上的平均MAE Loss: {mae_loss}")
    #     print(f"测试集上的准确率Accuracy: {accuracy}")
    #     print(f"测试集上的精确率Precision: {precision}")
    #     print(f"测试集上的召回率Recall: {recall}")
    #     print("DeepFM 测试完成!")
    #     return bce_loss, mae_loss, accuracy, precision, recall
    #
    # '''增量训练'''
    # def update_train(self, newly_update_time: str) -> None:
    #     """
    #     将新数据送进模型进行训练 训练完后同时更新id_mapping和model
    #     同一时刻最多同时增量训练一次
    #     :param newly_update_time: 该时间之后的新数据
    #     """
    #     mutex = threading.Lock()
    #     print('DeepFM 开始增量训练')
    #     try:
    #         mutex.acquire()
    #
    #         new_id_mapping, n_users, _, n_tags = self.__get_id_mapping__()  # 创建新的mapping
    #         new_model = MultiFeatureFM(n_users, n_tags, self.dim).to(self.device)  # 创建(embedding)更高维度的模型
    #         self.__load_model_parameters__(new_model)  # 加载已保存的参数到新模型
    #         new_dataloader = self.__get_full_dataloader__(newly_update_time)  # 加载新的dataloader
    #         if new_dataloader is None or len(new_dataloader) <= 0:
    #             print('DeepFM 已终止增量训练!')
    #             return
    #
    #         self.train(new_model, new_dataloader)  # 训练
    #
    #         self.model = new_model  # 将model和id_mapping拷贝回去
    #         self.id_mapping = new_id_mapping
    #         print('DeepFM 已完成增量训练!')
    #     finally:
    #         mutex.release()
    #
    #
    # '''预测'''
    # def predict(self, user_ids: list, book_ids: list) -> list:
    #     """
    #     预测 通过给的user列表, book列表给出预测评分(会将点击率*5转化为分数) 执行推荐算法的最小操作
    #     :param user_ids: user_id列表 len = n
    #     :param book_ids: book_id列表 len = n
    #     :return: 预测结果  len = n
    #     model=> outputs[k] == (user_id[k]->隐向量u[k] 与 book_id[k]->隐向量b[k]的内积)
    #     """
    #     user_ids, book_ids = self.id_mapping.mapping_ids(user_ids, book_ids)
    #     self.model.eval()
    #     with torch.no_grad():
    #         user_ids = torch.tensor(user_ids).long().to(self.device)
    #         tag_ids = self.__get_tags_by_book_id__(torch.tensor(book_ids)).to(self.device)
    #         outputs = 5 * self.model(user_ids, tag_ids)
    #     return outputs.tolist()
    #
    #
    # '''召回'''
    # def recall(self, user_id: int, k: int = 10, threshold: float = 4.5) -> list:
    #     """
    #     召回 选出k件目标用户可能评分最高的book 其中评分必须大于等于threshold
    #     :param user_id: 目标用户id
    #     :param k: 最高的k个
    #     :param threshold: 最低评分阈值
    #     :return: <=k行2列 list 第一列: book_id 第二列: 预测的rating
    #     """
    #     # 如果用户是新用户 返回空列表 回到java后端给推送热门book + 下次增量训练后再说
    #     if user_id not in self.id_mapping.user_id_map:
    #         return []
    #
    #     # 先预测
    #     book_ids = self.id_mapping.book_id_map.keys()
    #     user_ids = [user_id] * len(book_ids)
    #     ratings = self.predict(user_ids, book_ids)
    #     book_rating_pairs = [(book_id, rating) for book_id, rating in zip(book_ids, ratings)]
    #
    #     # 筛选出评分大于等于阈值的元组，并使用最大堆获取最高的 k 个评分
    #     heap = [item for item in book_rating_pairs if item[1] >= threshold]
    #     recall_list = heapq.nlargest(k, heap, key=lambda x: x[1])
    #     return recall_list
    #
    # '''top_k测试 基于预测结果的(全负)召回率与准确率测试'''
    # def test_top_k(self, min_rating_num: int = 200, k: int = 100, threshold: float = 4.0) -> tuple[float, float, float]:
    #     """
    #     :param min_rating_num: 最小发表的评价数量
    #     :param k: 选取k个作为推荐(召回)结果
    #     :param threshold: 最低评分阈值
    #     :return: avg_precision 精确度, avg_precision_full 全负精确度, avg_recall 召回率
    #     """
    #     return super(MultiFeatureFMSystem, self).test_top_k(min_rating_num, k, threshold)

    '''获取数据集'''
    def __get_dataset__(self, newly_update_time: str):
        return RatingDataset(newly_update_time)


    def __load_model_parameters__(self, model):
        """
        加载模型参数
         (将旧模型的参数复制到新模型中，保持行对齐)
        """
        if os.path.exists(self.params_saved_path):
            checkpoint = torch.load(self.params_saved_path)
            state_dict = checkpoint['model']
            model.users_embedding.weight.data[:len(state_dict['users_embedding.weight'])] = state_dict['users_embedding.weight']
            model.tags_embedding.weight.data[:len(state_dict['tags_embedding.weight'])] = state_dict['tags_embedding.weight']
            mlp_layers = list(model.mlp.children())   # 给 mlp 中的每层线性层分别赋值
            for i, layer in enumerate(mlp_layers):
                if isinstance(layer, nn.Linear):
                    # 赋值权重和偏置
                    model.mlp[i].weight.data = state_dict['mlp.' + str(i) + '.weight']
                    model.mlp[i].bias.data = state_dict['mlp.' + str(i) + '.bias']
        print('DeepFM 已加载模型参数') if os.path.exists(self.params_saved_path) else print('DeepFM 无已保存的模型参数')


if __name__ == '__main__':
    multi_feature_fm_system = MultiFeatureFMSystem()
    multi_feature_fm_system.train(dataloader=multi_feature_fm_system.train_dataloader)
    # # deep_fm_system.test(dataloader=deep_fm_system.test_dataloader)
    # for k in [10, 20, 50, 100, 200, 500, 1000, 2000]:
    #     print(f'=========== k: {k} ===========')
    #     deep_fm_system.test_top_k(min_rating_num=50, k=k, threshold=4.0)