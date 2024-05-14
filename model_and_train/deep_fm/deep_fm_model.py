from torch import nn
import torch



class DeepFM(nn.Module):
    def __init__(self, n_users, n_item_tags, dim):
        super(DeepFM, self).__init__()
        self.users_embedding = nn.Embedding(n_users, dim, max_norm=2)
        self.tags_embedding = nn.Embedding(n_item_tags, dim, max_norm=2)
        mlp_input_dim = dim * 9  # user_id + 8 * item_tag
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_input_dim // 2, mlp_input_dim // 4),
            nn.ReLU(),
            nn.Linear(mlp_input_dim // 4, 1),
            nn.Sigmoid()
        )

    def FMcross(self, features_emb):
        square_of_sum = torch.sum(features_emb, dim=1) ** 2   # shape: [batch_size, 9, dim] -> [batch_size, dim]
        sum_of_square = torch.sum(features_emb ** 2, dim=1)   # shape: [batch_size, 9, dim] -> [batch_size, dim]
        output = square_of_sum - sum_of_square
        output = torch.sum(output, dim=1, keepdim=True)  # shape = [batch_size, dim] -> [batch_size, 1]
        output = 0.5 * output
        output = torch.squeeze(output)  # shape = [batch_size, 1] -> [batch_size]
        return output

    def deep(self, features_emb):
        features_emb = features_emb.reshape([features_emb.shape[0], -1])  # shape: [batch_size, 9, dim] -> [batch_size, 9 * dim]
        output = self.mlp(features_emb)
        output = torch.squeeze(output)
        return output

    def forward(self, user_id, tag_ids):
        user_emb = self.users_embedding(user_id)    # shape: [batch_size, dim]
        user_emb_extended = torch.unsqueeze(user_emb, dim=1)    # shape: [batch_size, dim] -> [batch_size, 1, dim]
        item_tags_emb = self.tags_embedding(tag_ids)     # shape: [batch_size, 8, dim], 8个标签
        # 拼接 用户ID_emb 与 8个文学作品标签emb
        features_emb = torch.cat([user_emb_extended, item_tags_emb], dim=1)  # shape: [batch_size, 9, dim]

        fm_output = self.FMcross(features_emb)     # shape: [batch_size]
        mlp_output = self.deep(features_emb)        # shape: [batch_size]
        output = fm_output + mlp_output
        click_possibility = torch.sigmoid(output)
        return click_possibility  # 输出的是点击概率
