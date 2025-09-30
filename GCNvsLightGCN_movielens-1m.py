import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from torch_geometric.data import Data
from torch_geometric.nn import LightGCN, GCNConv
from torch_geometric.utils import degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== 加载MovieLens-1M数据集 =====================
def load_movielens_1m(data_path):
    """
    加载MovieLens-1M数据集并转换为PyG图数据格式
    """
    # 读取评分数据
    ratings = pd.read_csv(osp.join(data_path, 'ratings.dat'),
                          sep='::',
                          engine='python',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # 读取电影数据
    movies = pd.read_csv(osp.join(data_path, 'movies.dat'),
                         sep='::',
                         engine='python',
                         names=['movie_id', 'title', 'genres'],
                         encoding='latin-1')

    # 读取用户数据
    users = pd.read_csv(osp.join(data_path, 'users.dat'),
                        sep='::',
                        engine='python',
                        names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])

    print(f"数据集统计: {len(ratings)} 条评分, {len(users)} 个用户, {len(movies)} 部电影")

    # 重新映射用户和电影ID为连续整数
    user_mapping = {orig: new for new, orig in enumerate(ratings['user_id'].unique())}
    movie_mapping = {orig: new for new, orig in enumerate(ratings['movie_id'].unique())}

    ratings['user_idx'] = ratings['user_id'].map(user_mapping)
    ratings['movie_idx'] = ratings['movie_id'].map(movie_mapping)

    num_users = len(user_mapping)
    num_movies = len(movie_mapping)

    print(f"重新映射后: {num_users} 个用户, {num_movies} 部电影")

    return ratings, num_users, num_movies


# 加载数据
data_path = './datasets/ml-1m/'  # 请修改为你的MovieLens-1M数据路径
ratings, num_users, num_movies = load_movielens_1m(data_path)

# ===================== 数据划分 (8:1:1) =====================
# 首先划分训练集和临时测试集 (8:2)
train_df, temp_df = train_test_split(ratings, test_size=0.2, random_state=42)
# 然后将临时测试集划分为验证集和测试集 (1:1)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"训练集: {len(train_df)} 条评分")
print(f"验证集: {len(val_df)} 条评分")
print(f"测试集: {len(test_df)} 条评分")


# ===================== 构建图数据 =====================
def build_edge_index(df, num_users, num_movies):
    """
    从DataFrame构建边索引
    """
    # 用户节点ID: 0 到 num_users-1
    # 电影节点ID: num_users 到 num_users + num_movies - 1
    user_nodes = torch.tensor(df['user_idx'].values, dtype=torch.long)
    movie_nodes = torch.tensor(df['movie_idx'].values, dtype=torch.long) + num_users

    # 构建双向边 (用户-电影 和 电影-用户)
    edge_index = torch.stack([
        torch.cat([user_nodes, movie_nodes]),  # 源节点
        torch.cat([movie_nodes, user_nodes])  # 目标节点
    ], dim=0)

    return edge_index


# 构建训练图的边索引
train_edge_index = build_edge_index(train_df, num_users, num_movies)

# 构建验证和测试的边标签索引
val_edge_label_index = build_edge_index(val_df, num_users, num_movies)
test_edge_label_index = build_edge_index(test_df, num_users, num_movies)

# 创建PyG Data对象
data = Data(
    edge_index=train_edge_index,
    num_nodes=num_users + num_movies
).to(device)

# 为验证和测试集存储边标签索引
data.val_edge_label_index = val_edge_label_index.to(device)
data.test_edge_label_index = test_edge_label_index.to(device)

# ===================== 准备训练数据 =====================
batch_size = 8192
# 使用所有训练边作为正样本，但要去重（因为构建了双向边）
mask = data.edge_index[0] < data.edge_index[1]
train_edge_label_index = data.edge_index[:, mask]
train_loader = torch.utils.data.DataLoader(
    range(train_edge_label_index.size(1)),
    shuffle=True,
    batch_size=batch_size,
)


# ===================== 基于GCN的推荐模型 =====================
class GCNRecommender(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

    def forward(self, edge_index):
        x = self.embedding.weight
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def get_embedding(self, edge_index):
        return self.forward(edge_index)

    def recommendation_loss(self, pos_score, neg_score):
        return -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-15).mean()


# ===================== 公共训练与测试函数 =====================
def train_one_epoch(model, optimizer, train_loader, edge_index, train_edge_label_index, num_users, num_movies):
    total_loss = total_examples = 0
    model.train()

    for index in tqdm(train_loader, leave=False):
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_users, num_users + num_movies,
                          (index.numel(),), device=device)
        ], dim=0)
        edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)

        optimizer.zero_grad()
        emb = model.get_embedding(edge_index)
        pos_rank, neg_rank = (emb[edge_label_index[0]] * emb[edge_label_index[1]]).sum(dim=-1).chunk(2)
        loss = model.recommendation_loss(pos_rank, neg_rank)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(model, edge_index, test_edge_label_index, train_edge_label_index, num_users, num_movies, batch_size,
         k: int = 20):
    model.eval()
    emb = model.get_embedding(edge_index)
    user_emb, movie_emb = emb[:num_users], emb[num_users:num_users + num_movies]

    precision = recall = total_examples = 0

    # 获取测试集中的所有用户
    test_users = test_edge_label_index[0, test_edge_label_index[0] < num_users].unique()

    for start in range(0, len(test_users), batch_size):
        end = min(start + batch_size, len(test_users))
        user_batch = test_users[start:end]

        # 计算当前批次用户与所有电影的相似度
        logits = user_emb[user_batch] @ movie_emb.t()

        # 排除训练边
        for i, user in enumerate(user_batch):
            # 找到该用户在训练集中的所有交互
            mask = (train_edge_label_index[0] == user)
            trained_movies = train_edge_label_index[1, mask] - num_users
            if len(trained_movies) > 0:
                logits[i, trained_movies] = float('-inf')

        # 创建真实标签矩阵
        ground_truth = torch.zeros((len(user_batch), num_movies), dtype=torch.bool, device=device)
        for i, user in enumerate(user_batch):
            # 找到该用户在测试集中的所有交互
            mask = (test_edge_label_index[0] == user) & (test_edge_label_index[1] >= num_users)
            test_movies = test_edge_label_index[1, mask] - num_users
            if len(test_movies) > 0:
                ground_truth[i, test_movies] = True

        # 计算每个用户的测试边数量
        node_count = ground_truth.sum(dim=1)

        # 获取Top-K预测
        topk_index = logits.topk(k, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)

        # 计算精确率和召回率
        precision += float((isin_mat.sum(dim=-1) / k).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples


# ===================== 训练 LightGCN 和 GCN 对比 =====================
def run_experiment(model_class, name, epochs=100, lr=0.001):
    if name == "LightGCN":
        model = LightGCN(num_nodes=data.num_nodes, embedding_dim=64, num_layers=2).to(device)
    else:
        model = GCNRecommender(num_nodes=data.num_nodes, embedding_dim=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_precisions, val_recalls = [], [], []
    test_precisions, test_recalls = [], []

    for epoch in range(1, epochs + 1):
        # 训练
        loss = train_one_epoch(model, optimizer, train_loader, data.edge_index,
                               train_edge_label_index, num_users, num_movies)

        # 在验证集上测试
        val_precision, val_recall = test(model, data.edge_index, data.val_edge_label_index,
                                         train_edge_label_index, num_users, num_movies, batch_size, k=20)

        # 在测试集上测试（每10个epoch测试一次）
        if epoch % 10 == 0:
            test_precision, test_recall = test(model, data.edge_index, data.test_edge_label_index,
                                               train_edge_label_index, num_users, num_movies, batch_size, k=20)
            test_precisions.append(test_precision)
            test_recalls.append(test_recall)
        else:
            test_precision, test_recall = 0, 0

        train_losses.append(loss)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

        print(f"[{name}] Epoch {epoch:03d} | Loss: {loss:.4f} | "
              f"Val P@20: {val_precision:.4f} | Val R@20: {val_recall:.4f}"
              f"{f' | Test P@20: {test_precision:.4f} | Test R@20: {test_recall:.4f}' if epoch % 10 == 0 else ''}")

    return train_losses, val_precisions, val_recalls, test_precisions, test_recalls


# ===================== 主流程 =====================
epochs = 100
print("开始训练 LightGCN...")
lgn_loss, lgn_val_p, lgn_val_r, lgn_test_p, lgn_test_r = run_experiment(LightGCN, "LightGCN", epochs=epochs)

print("\n开始训练 GCN...")
gcn_loss, gcn_val_p, gcn_val_r, gcn_test_p, gcn_test_r = run_experiment(GCNRecommender, "GCN", epochs=epochs)

# ===================== 可视化对比 =====================
plt.figure(figsize=(18, 5))

# 训练损失
plt.subplot(1, 4, 1)
plt.plot(lgn_loss, label='LightGCN')
plt.plot(gcn_loss, label='GCN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# 验证集 Precision@20
plt.subplot(1, 4, 2)
plt.plot(lgn_val_p, label='LightGCN', color='blue')
plt.plot(gcn_val_p, label='GCN', color='red')
plt.xlabel('Epoch')
plt.ylabel('Precision@20')
plt.title('Validation Precision@20')
plt.legend()

# 验证集 Recall@20
plt.subplot(1, 4, 3)
plt.plot(lgn_val_r, label='LightGCN', color='blue')
plt.plot(gcn_val_r, label='GCN', color='red')
plt.xlabel('Epoch')
plt.ylabel('Recall@20')
plt.title('Validation Recall@20')
plt.legend()

# 测试集指标（稀疏点）
plt.subplot(1, 4, 4)
test_epochs = list(range(10, epochs + 1, 10))
plt.plot(test_epochs, lgn_test_p, 'o-', label='LightGCN Precision@20', color='blue')
plt.plot(test_epochs, lgn_test_r, 's-', label='LightGCN Recall@20', color='lightblue')
plt.plot(test_epochs, gcn_test_p, 'o-', label='GCN Precision@20', color='red')
plt.plot(test_epochs, gcn_test_r, 's-', label='GCN Recall@20', color='pink')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Test Metrics (every 10 epochs)')
plt.legend()

plt.tight_layout()
plt.savefig("movielens_lgn_vs_gcn.png", dpi=300, bbox_inches='tight')
plt.show()

# 打印最终测试结果
print("\n=== 最终测试结果 ===")
print(f"LightGCN - 最终测试 Precision@20: {lgn_test_p[-1]:.4f}, Recall@20: {lgn_test_r[-1]:.4f}")
print(f"GCN - 最终测试 Precision@20: {gcn_test_p[-1]:.4f}, Recall@20: {gcn_test_r[-1]:.4f}")