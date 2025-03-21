import os.path

from model_and_train.multi_feature_fm.multi_feature_fm_trainer_and_predictor import MultiFeatureFMSystem

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
    bert_path = os.path.join(script_dir, os.path.join("model_and_train", "multi_feature_fm", "base_bert_chinese"))
    multi_feature_fm_system = MultiFeatureFMSystem(bert_version=bert_path, epochs=5)
    multi_feature_fm_system.train(dataloader=multi_feature_fm_system.train_dataloader)
    multi_feature_fm_system.test(dataloader=multi_feature_fm_system.test_dataloader)