# ===================== 导入库 =====================
import os.path as osp
import torch
from tqdm import tqdm

import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import LightGCN
from torch_geometric.utils import degree

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ===================== 设置设备 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== 加载 MovieLens-1M 数据 =====================
path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets', 'ml-1m')
ratings_path = osp.join(path, "ratings.dat")

ratings = pd.read_csv(ratings_path, sep="::", engine="python",
                      names=["userId", "movieId", "rating", "timestamp"])

# 转换为隐式反馈：评分 >=4 视为正样本
ratings = ratings[ratings["rating"] >= 4]

# 用户 & 电影重新映射为连续索引
user_id_map = {id: i for i, id in enumerate(ratings["userId"].unique())}
movie_id_map = {id: i for i, id in enumerate(ratings["movieId"].unique())}
ratings["userId"] = ratings["userId"].map(user_id_map)
ratings["movieId"] = ratings["movieId"].map(movie_id_map)

num_users = len(user_id_map)
num_movies = len(movie_id_map)


# ===================== 划分训练/验证/测试集 (8:1:1) =====================
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

def build_edge_index(df):
    return torch.tensor([
        df["userId"].values,
        df["movieId"].values + num_users
    ], dtype=torch.long)

train_edge_index = build_edge_index(train_df)
valid_edge_index = build_edge_index(valid_df)
test_edge_index = build_edge_index(test_df)


# ===================== 构建HeteroData图 =====================
data = HeteroData()
data["user"].num_nodes = num_users
data["movie"].num_nodes = num_movies
data["user", "rates", "movie"].edge_index = train_edge_index
data = ToUndirected()(data)
data = data.to_homogeneous().to(device)


# ===================== 初始化 =====================
batch_size = 8192
train_losses, val_precisions, val_recalls, test_precisions, test_recalls = [], [], [], [], []


# ===================== 训练数据加载器 =====================
mask = data.edge_index[0] < data.edge_index[1]
train_edge_label_index = data.edge_index[:, mask]
train_loader = torch.utils.data.DataLoader(
    range(train_edge_label_index.size(1)),
    shuffle=True,
    batch_size=batch_size,
)


# ===================== 初始化模型和优化器 =====================
model = LightGCN(
    num_nodes=data.num_nodes,
    embedding_dim=64,
    num_layers=2,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ===================== 训练函数 =====================
def train():
    total_loss = total_examples = 0
    for index in tqdm(train_loader):
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_users, num_users + num_movies,
                          (index.numel(),), device=device)
        ], dim=0)
        edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)

        optimizer.zero_grad()
        pos_rank, neg_rank = model(data.edge_index, edge_label_index).chunk(2)
        loss = model.recommendation_loss(
            pos_rank, neg_rank, node_id=edge_label_index.view(-1).unique()
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()
    return total_loss / total_examples


# ===================== 评估函数 =====================
@torch.no_grad()
def evaluate(edge_index, k=20):
    emb = model.get_embedding(data.edge_index)
    user_emb, movie_emb = emb[:num_users], emb[num_users:]

    precision = recall = total_examples = 0
    for start in range(0, num_users, batch_size):
        end = start + batch_size
        logits = user_emb[start:end] @ movie_emb.t()

        # 去掉训练边
        mask = ((train_edge_label_index[0] >= start) &
                (train_edge_label_index[0] < end))
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - num_users] = float('-inf')

        # ground truth = edge_index (valid 或 test)
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((edge_index[0] >= start) & (edge_index[0] < end))
        ground_truth[edge_index[0, mask] - start,
                     edge_index[1, mask] - num_users] = True
        node_count = degree(edge_index[0, mask] - start, num_nodes=logits.size(0))

        topk_index = logits.topk(k, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)

        precision += float((isin_mat.sum(dim=-1) / k).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples


# ===================== 训练循环 =====================
for epoch in range(1, 51):
    loss = train()
    val_precision, val_recall = evaluate(valid_edge_index.to(device))
    test_precision, test_recall = evaluate(test_edge_index.to(device))

    train_losses.append(loss)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    test_precisions.append(test_precision)
    test_recalls.append(test_recall)

    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
          f"Val_P@20: {val_precision:.4f}, Val_R@20: {val_recall:.4f}, "
          f"Test_P@20: {test_precision:.4f}, Test_R@20: {test_recall:.4f}")


# ===================== 可视化 =====================
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss"); plt.legend()

plt.subplot(1, 3, 2)
plt.plot(val_precisions, label="Val Precision@20")
plt.plot(test_precisions, label="Test Precision@20")
plt.xlabel("Epoch"); plt.ylabel("Precision"); plt.title("Precision@20"); plt.legend()

plt.subplot(1, 3, 3)
plt.plot(val_recalls, label="Val Recall@20")
plt.plot(test_recalls, label="Test Recall@20")
plt.xlabel("Epoch"); plt.ylabel("Recall"); plt.title("Recall@20"); plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.close()
