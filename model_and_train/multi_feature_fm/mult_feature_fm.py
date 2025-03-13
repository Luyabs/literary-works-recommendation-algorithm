import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class MultiFeatureFM(nn.Module):
    def __init__(self, n_users, n_item_tags, dim, bert_version):
        super(MultiFeatureFM, self).__init__()

        # Embedding 层
        self.users_embedding = nn.Embedding(n_users, dim, max_norm=2)
        self.tags_embedding = nn.Embedding(n_item_tags, dim, max_norm=2)

        # BERT (冻结参数）
        self.bert_encoder = BertModel.from_pretrained(bert_version, local_files_only=True)
        for param in self.bert_encoder.parameters():
            param.requires_grad = False  # 冻结 BERT

        # BERT 输出降维层
        self.text_fc = nn.Linear(self.bert_encoder.config.hidden_size, dim)

        # Deep 部分 MLP
        mlp_input_dim = dim * 3  # user_id + item_description + ave(item_tag)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_input_dim // 2, mlp_input_dim // 4),
            nn.ReLU(),
            nn.Linear(mlp_input_dim // 4, 1),
            nn.Sigmoid()
        )

    def FMcross(self, features_emb):
        square_of_sum = torch.sum(features_emb, dim=1) ** 2
        sum_of_square = torch.sum(features_emb ** 2, dim=1)
        output = square_of_sum - sum_of_square
        output = torch.sum(output, dim=1, keepdim=True)  # shape: [batch_size, 1]
        return 0.5 * torch.squeeze(output)

    def deep(self, features_emb):
        features_emb = features_emb.reshape([features_emb.shape[0], -1])  # shape: [batch_size, 3 * dim]
        output = self.mlp(features_emb)
        return torch.squeeze(output)


    def encode_text(self, input_ids, attention_mask):
        """ 用 BERT 提取文本特征 """
        with torch.no_grad():
            bert_output = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = bert_output.last_hidden_state[:, 0, :]  # 取 CLS 位置的向量
        return self.text_fc(text_emb)  # 映射到 dim

    def forward(self, user_id, tag_ids, item_text_input, item_text_mask, tag_text_input, tag_text_mask):
        """
        user_id: 用户ID
        tag_ids: item tags ID
        book_text_input, book_text_mask: 书本文本的 tokenized input_ids 和 attention_mask
        tag_text_input, tag_text_mask: 标签文本的 tokenized input_ids 和 attention_mask
        """
        user_emb = self.users_embedding(user_id)  # shape: [batch_size, dim]

        item_tags_emb = self.tags_embedding(tag_ids)  # shape: [batch_size, 8, dim]
        item_tags_emb = torch.mean(item_tags_emb, dim=1)  # shape: [batch_size, dim]

        # 获取书本文本特征
        item_text_emb = self.encode_text(item_text_input, item_text_mask)  # shape: [batch_size, dim]

        # 拼接 用户ID_emb + 书本文本 + 8个 item_tag
        features_emb = torch.stack([user_emb, item_text_emb, item_tags_emb],
                                 dim=1)  # shape: [batch_size, 3, dim]

        fm_output = self.FMcross(features_emb)
        mlp_output = self.deep(features_emb)
        output = fm_output + mlp_output
        click_possibility = torch.sigmoid(output)

        # 计算对比损失（BERT tag vs. Embedding tag）
        tag_text_emb = self.encode_text(tag_text_input, tag_text_mask)  # shape: [batch_size, dim]
        contrastive_loss = F.mse_loss(tag_text_emb, item_text_emb)  # 均方误差

        return click_possibility, contrastive_loss
