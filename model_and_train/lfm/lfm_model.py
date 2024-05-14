import torch
from torch import nn

class LFM(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super(LFM, self).__init__()
        self.users_embedding = nn.Embedding(n_users, dim, max_norm=2)
        self.items_embedding = nn.Embedding(n_items, dim, max_norm=2)

    def forward(self, user_id, item_id):
        user_emb = self.users_embedding(user_id)    # shape: [batch_size, dim]
        item_emb = self.items_embedding(item_id)    # shape: [batch_size, dim]
        output = (user_emb * item_emb).sum(dim=1)   # shape: [batch_size, dim] -> [batch_size]
        rating_hat = 5 * torch.sigmoid(output)  # 将输出映射到 [0, 5] 范围内
        return rating_hat  # 输出的预计评分
