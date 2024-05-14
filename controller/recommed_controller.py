import random
import threading

from flask import Flask, request, jsonify
from tqdm import tqdm

from mapper.train_and_predictor_mapper import select_user_having_ratings_greater_equal, select_rating_by_user_id
from model_and_train.fm.fm_trainer_and_predictor import FmSystem
from model_and_train.lfm.lfm_trainer_and_predictor import LfmSystem


class BackendFlask():
    def __init__(self) -> None:
        super().__init__()
        self.app = Flask(__name__)
        self.train_lock = threading.Lock()  # init_train操作与update_train操作的互斥锁
        self.mutex = threading.Lock()  # 单步操作的互斥锁
        self.lfm_system = LfmSystem(epochs=20, lr=0.002)  # 初始化模型系统
        self.fm_system = FmSystem(epochs=20, lr=0.002)  # 初始化模型系统

        @self.app.route('/recommend/is_training', methods=['GET'])
        def is_training():
            return jsonify({'is_training': self.train_lock.locked()})

        @self.app.route('/recommend/update_train', methods=['POST'])
        def update_train():
            """
            增量训练的API
            接收 POST 请求，包含 newly_update_time 作为 JSON 格式的数据
            执行增量训练并返回结果
            """
            try:
                self.mutex.acquire()
                if self.train_lock.locked():
                    return jsonify({'message': '同一时间只允许一个线程执行 初始化训练/增量训练'}), 200
                else:
                    self.train_lock.acquire()
            finally:
                self.mutex.release()

            req_data = request.get_json()
            newly_update_time = req_data.get('newly_update_time')
            if not newly_update_time:
                return jsonify({'error': 'newly_update_time is required'}), 400

            try:
                self.lfm_system.update_train(newly_update_time)
                self.fm_system.update_train(newly_update_time)
                return jsonify({'message': 'Incremental training completed successfully'}), 200
            except Exception as e:
                print(type(e), e)
                return jsonify({'error_type': str(type(e)),
                                'error': str(e)}), 400
            finally:
                self.train_lock.release()

        @self.app.route('/recommend/predict', methods=['POST'])
        def predict():
            """
            预测评分的API
            接收 POST 请求，包含 user_ids 和 book_ids 作为 JSON 格式的数据
            返回预测评分的结果
            """
            req_data = request.get_json()
            user_ids = req_data.get('user_ids', [])
            book_ids = req_data.get('book_ids', [])
            if not user_ids or not book_ids:
                return jsonify({'error': 'user_ids and book_ids are required'}), 400

            try:
                predictions = self.fm_system.predict(user_ids, book_ids)
                return jsonify({'predictions': predictions}), 200
            except Exception as e:
                print(type(e), e)
                return jsonify({'error_type': str(type(e)),
                                'error': str(e)}), 400

        @self.app.route('/recommend/recall', methods=['POST'])
        def recall():
            """
            召回推荐的API
            接收 POST 请求，包含 user_id、k、threshold、 mix_weight(两个模型混合评分) 作为 JSON 格式的数据
            返回召回的推荐列表
            """
            req_data = request.get_json()
            user_id = req_data.get('user_id')
            if user_id is None:
                return jsonify({'error': 'user_id is required'}), 400

            k = req_data.get('k', 10)  # 默认为 10
            threshold = req_data.get('threshold', 4.0)  # 默认为 4.0
            mix_weight = req_data.get('mix_weight', 0.1)  # 默认为 0.1

            try:
                # 由LFM先recall min(2000, k * 75)
                lfm_recall_list = self.lfm_system.recall(user_id, min(2000, k * 75), threshold)
                if not lfm_recall_list:
                    return jsonify({'recommend_list': []}), 200

                # 再由FM predict 这些 item
                book_ids = [book_id[0] for book_id in lfm_recall_list]
                user_ids = [user_id] * len(book_ids)
                fm_predict_list = self.fm_system.predict(user_ids, book_ids)
                # 把LFM的输出结果 与 FM的输出结果加权求和
                combined_recommend_list = [
                    [lfm_recall_list[0], lfm_recall_list[1] * mix_weight + fm_score * (1 - mix_weight)]
                    for lfm_recall_list, fm_score in zip(lfm_recall_list, fm_predict_list)
                ]
                # 筛选出一部分 (5 * k个)
                combined_recommend_list = sorted(combined_recommend_list, reverse=True)[:5 * k]
                # 根据依分数 依概率返回最后结果
                final_select_items = random.choices(combined_recommend_list,
                                                    weights=[score if score >= threshold else 0
                                                             for _, score in combined_recommend_list], k=k)
                # 去除 final_select_items 重复项
                final_select_ids = set()
                unique_final_select_items = []
                for index, (book_id, score) in enumerate(final_select_items):
                    if book_id not in final_select_ids:
                        final_select_ids.add(book_id)
                        unique_final_select_items.append([book_id, score])
                return jsonify({'recommend_list': unique_final_select_items}), 200
            except Exception as e:
                print(type(e), e)
                return jsonify({'error_type': str(type(e)),
                                'error': str(e)}), 400

        @self.app.route('/recommend/similar', methods=['POST'])
        def get_similar_book():
            """
            根据LFM推荐 与 指定book有相似交互行为的其它book
            接收 POST 请求，包含 book_id 和 k 作为 JSON 格式的数据
            返回预测评分的结果
            """
            req_data = request.get_json()
            book_id = req_data.get('book_id')
            if book_id is None:
                return jsonify({'error': 'book_id is required'}), 400
            k = req_data.get('k', 10)  # 默认为 10

            try:
                # 找出前 5 * k 个相似物品
                similar_books = self.lfm_system.calculate_similarity(book_id, k * 5)
                if not similar_books:
                    return jsonify({'recommend_list': []}), 200

                # 根据依分数 依概率返回最后结果
                final_select_books = random.choices(similar_books,
                                                    weights=[similarity for _, similarity in similar_books], k=k)
                # 去除 final_select_items 重复项
                final_select_ids = set()
                unique_similar_books = []
                for index, (book_id, score) in enumerate(final_select_books):
                    if book_id not in final_select_ids:
                        final_select_ids.add(book_id)
                        unique_similar_books.append([book_id, score])
                return jsonify({'recommend_list': unique_similar_books}), 200
            except Exception as e:
                print(type(e), e)
                return jsonify({'error_type': str(type(e)),
                                'error': str(e)}), 400

        '''top_k测试 基于预测结果的(全负)召回率与准确率测试'''
        ''''''
        @self.app.route('/recommend/top-K', methods=['POST'])
        def test_top_k(self):
            print("基于预测结果的top-K测试开始")
            req_data = request.get_json()
            min_rating_num = req_data.get('min_rating_num', 200)
            k = req_data.get('k', 10)
            threshold = req_data.get('threshold', 4.5)
            mix_weight = req_data.get('mix_weight', 0.8)
            recall_ratio = req_data.get('recall_ratio', 10)
            """
            :param min_rating_num: 最小发表的评价数量
            :param k: 选取k个作为推荐(召回)结果
            :param threshold: 最低评分阈值
            :mix_weight: 
            :return: avg_precision 精确度, avg_precision_full 全负精确度, avg_recall 召回率
            """

            try:
                user_ids = select_user_having_ratings_greater_equal(min_rating_num)
                avg_precision = 0
                avg_precision_full = 0
                avg_recall = 0

                for user_id in tqdm(user_ids):
                    # 由LFM先recall (recall_ratio * k个)
                    lfm_recall_list = self.lfm_system.recall(user_id, recall_ratio * k, threshold)

                    # 再由FM/Deep FM predict 这些 item
                    book_ids = [book_id[0] for book_id in lfm_recall_list]
                    user_ids = [user_id] * len(book_ids)
                    fm_predict_list = self.fm_system.predict(user_ids, book_ids)
                    # 把LFM的输出结果 与 FM的输出结果加权求和
                    combined_recommend_list = [
                        [lfm_recall_list[0], lfm_recall_list[1] * mix_weight + fm_score * (1 - mix_weight)]
                        for lfm_recall_list, fm_score in zip(lfm_recall_list, fm_predict_list)
                    ]
                    pred = {book_id for book_id, rating in combined_recommend_list}  # pred是预测结果
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

                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision 准确率 = TP / (TP + FP)
                    precision_full = TP / len(pred) if len(pred) > 0 else 0  # Precision_Full 全负准确率 = TP / (TP + FP)
                    recall = TP / P if P > 0 else 0  # Recall 召回率 = TP / P
                    avg_precision += precision
                    avg_precision_full += precision_full
                    avg_recall += recall

                avg_precision /= len(user_ids)
                avg_precision_full /= len(user_ids)
                avg_recall /= len(user_ids)
                print(f"预测列表上的top-K精确率: {avg_precision}")
                print(f"预测列表上的top-K全负准确率: {avg_precision_full}")
                print(f"预测列表上的top-K召回率: {avg_recall}")
                print("基于预测结果的top-K测试结束!")
                return jsonify({'avg_precision': avg_precision, 'avg_precision_full': avg_precision_full, 'avg_recall': avg_recall}), 200
            except Exception as e:
                print(type(e), e)
                return jsonify({'error_type': str(type(e)),
                                'error': str(e)}), 400

        # @self.app.route('/lfm/test/dataset', methods=['POST'])
        # def test():
        #     """
        #     测试模型的API
        #     接收 POST 请求，无需参数
        #     返回模型在测试集上的平均Loss、准确率、精确率与召回率
        #     """
        #     try:
        #         mse_loss, mae_loss, accuracy, precision, recall = self.lfm_system.test()
        #         return jsonify({'message': 'Model testing completed successfully',
        #                         'mse_loss': float(mse_loss),
        #                         'mae_loss': float(mae_loss),
        #                         'accuracy': float(accuracy),
        #                         'precision': float(precision),
        #                         'recall': float(recall)
        #                         }), 200
        #     except Exception as e:
        #         print(type(e), e)
        #         return jsonify({'error_type': str(type(e)),
        #                         'error': str(e)}), 400
        #
        #
        # @self.app.route('/lfm/test/prediction', methods=['POST'])
        # def test_top_k():
        #     """
        #     测试模型的API
        #     接收 POST 请求，需要参数 min_rating_num, k, threshold
        #     返回模型在实际预测结果上的精确率、全负精确率与召回率
        #     """
        #     req_data = request.get_json()
        #     min_rating_num = req_data.get('min_rating_num', 200)  # 默认为 200
        #     k = req_data.get('k', 1000)  # 默认为 1000
        #     threshold = req_data.get('threshold', 4.0)  # 默认为 4.0
        #
        #     try:
        #         precision, precision_full, recall = self.lfm_system.test_top_k(min_rating_num, k, threshold)
        #         return jsonify({'message': 'Model testing completed successfully',
        #                         'precision': float(precision),
        #                         'precision_full': float(precision_full),
        #                         'recall': float(recall)
        #                         }), 200
        #     except Exception as e:
        #         print(type(e), e)
        #         return jsonify({'error_type': str(type(e)),
        #                         'error': str(e)}), 400
        #
        # @self.app.route('/lfm/init_train', methods=['POST'])
        # def init_train():
        #     """
        #     模型训练的API
        #     接收 POST 请求，无需参数
        #     执行模型训练并返回结果
        #     """
        #     try:
        #         self.mutex.acquire()
        #         if self.train_lock.locked():
        #             return jsonify({'message': '同一时间只允许一个线程执行 初始化训练/增量训练'}), 200
        #         else:
        #             self.train_lock.acquire()
        #     finally:
        #         self.mutex.release()
        #
        #     try:
        #         self.lfm_system.train()
        #         return jsonify({'message': 'Model training completed successfully'}), 200
        #     except Exception as e:
        #         print(type(e), e)
        #         return jsonify({'error_type': str(type(e)),
        #                         'error': str(e)}), 400
        #     finally:
        #         self.train_lock.release()


if __name__ == '__main__':
    backend = BackendFlask()
    backend.app.run(port=8000)
