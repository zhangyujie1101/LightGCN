# ===================== 导入库 =====================
import os.path as osp
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.nn import LightGCN
from torch_geometric.data import Data

# ===================== 设置设备 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== 加载MovieLens-1M数据集 =====================
def load_data(data_path):
    ratings = pd.read_csv(osp.join(data_path, 'ratings.dat'),
                          sep='::', engine='python',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies = pd.read_csv(osp.join(data_path, 'movies.dat'),
                         sep='::', engine='python',
                         names=['movie_id', 'title', 'genres'],
                         encoding='latin-1')
    users = pd.read_csv(osp.join(data_path, 'users.dat'),
                        sep='::', engine='python',
                        names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
    print(f"数据集统计: {len(ratings)} 条评分, {len(users)} 个用户, {len(movies)} 部电影")

    user_mapping = {orig: new for new, orig in enumerate(ratings['user_id'].unique())}
    movie_mapping = {orig: new for new, orig in enumerate(ratings['movie_id'].unique())}

    ratings['user_idx'] = ratings['user_id'].map(user_mapping)
    ratings['movie_idx'] = ratings['movie_id'].map(movie_mapping)

    num_users = len(user_mapping)
    num_movies = len(movie_mapping)
    print(f"重新映射后: {num_users} 个用户, {num_movies} 部电影")

    return ratings, num_users, num_movies


data_path = './datasets/ml-1m/'
ratings, num_users, num_movies = load_data(data_path)

# ===================== 数据划分 (8:1:1) =====================
train_df, temp_df = train_test_split(ratings, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")

# ===================== 辅助函数 =====================
def build_edge_index(df, num_users, num_movies):
    """构建 LightGCN 边索引"""
    user_nodes = torch.tensor(df['user_idx'].values, dtype=torch.long)
    movie_nodes = torch.tensor(df['movie_idx'].values + num_users, dtype=torch.long)
    edge_index = torch.stack([torch.cat([user_nodes, movie_nodes]),
                              torch.cat([movie_nodes, user_nodes])], dim=0)
    return edge_index

train_edge_index = build_edge_index(train_df, num_users, num_movies).to(device)
data = Data(edge_index=train_edge_index, num_nodes=num_users + num_movies).to(device)

# ===================== 模型 1：MLP 双塔模型 =====================
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.movie_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, user_idx, movie_idx):
        user_vec = self.user_mlp(self.user_emb(user_idx))
        movie_vec = self.movie_mlp(self.movie_emb(movie_idx))
        return (user_vec * movie_vec).sum(dim=-1)

    def get_user_embedding(self):
        return self.user_mlp(self.user_emb.weight)

    def get_movie_embedding(self):
        return self.movie_mlp(self.movie_emb.weight)

# ===================== 模型 2：LightGCN =====================
class LightGCNModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64, num_layers=2):
        super().__init__()
        self.gcn = LightGCN(num_nodes=num_users + num_movies,
                            embedding_dim=embedding_dim,
                            num_layers=num_layers)
        self.num_users = num_users
        self.num_movies = num_movies

    def get_user_embedding(self, edge_index):
        emb = self.gcn.get_embedding(edge_index)
        return emb[:self.num_users]

    def get_movie_embedding(self, edge_index):
        emb = self.gcn.get_embedding(edge_index)
        return emb[self.num_users:self.num_users + self.num_movies]

# ===================== 公共函数：训练 & 测试 =====================
def train_mlp(model, optimizer, train_df, num_movies, batch_size):
    model.train()
    total_loss, total_examples = 0, 0
    loader = torch.utils.data.DataLoader(range(len(train_df)), shuffle=True, batch_size=batch_size)

    for idx in tqdm(loader, leave=False):
        batch = train_df.iloc[idx]
        user_idx = torch.tensor(batch['user_idx'].values, dtype=torch.long, device=device)
        movie_idx = torch.tensor(batch['movie_idx'].values, dtype=torch.long, device=device)
        neg_movie_idx = torch.randint(0, num_movies, (len(idx),), device=device)

        optimizer.zero_grad()
        pos_score = model(user_idx, movie_idx)
        neg_score = model(user_idx, neg_movie_idx)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * len(idx)
        total_examples += len(idx)

    return total_loss / total_examples


def train_lightgcn(model, optimizer, train_df, edge_index, num_users, num_movies, batch_size):
    model.train()
    total_loss, total_examples = 0, 0
    mask = edge_index[0] < edge_index[1]
    train_edge_label_index = edge_index[:, mask]
    loader = torch.utils.data.DataLoader(range(train_edge_label_index.size(1)),
                                         shuffle=True, batch_size=batch_size)

    with torch.no_grad():
        emb = model.gcn.get_embedding(edge_index)

    for index in tqdm(loader, leave=False):
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_users, num_users + num_movies, (index.numel(),), device=device)
        ], dim=0)

        optimizer.zero_grad()

        pos_score = (emb[pos_edge_label_index[0]] * emb[pos_edge_label_index[1]]).sum(dim=-1)
        neg_score = (emb[neg_edge_label_index[0]] * emb[neg_edge_label_index[1]]).sum(dim=-1)

        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * len(index)
        total_examples += len(index)

    return total_loss / total_examples


@torch.no_grad()
def test(eval_df, train_df, user_emb, movie_emb, k=10):
    precision, recall, ndcg = 0, 0, 0
    users = torch.tensor(eval_df['user_idx'].unique(), device=device)

    for start in range(0, len(users), 512):
        end = min(start + 512, len(users))
        batch_users = users[start:end]
        logits = user_emb[batch_users] @ movie_emb.t()

        # 屏蔽训练中看过的电影
        for i, u in enumerate(batch_users):
            train_movies = train_df[train_df['user_idx'] == u.item()]['movie_idx'].values
            logits[i, train_movies] = -float('inf')

        # 真实交互
        ground_truth = torch.zeros((len(batch_users), movie_emb.size(0)), dtype=torch.bool, device=device)
        for i, u in enumerate(batch_users):
            gt_movies = eval_df[eval_df['user_idx'] == u.item()]['movie_idx'].values
            ground_truth[i, gt_movies] = True

        topk = logits.topk(k, dim=-1).indices
        hits = ground_truth.gather(1, topk)

        precision += (hits.sum(dim=-1) / k).sum().item()
        recall += (hits.sum(dim=-1) / ground_truth.sum(dim=-1).clamp(1e-6)).sum().item()

        for i in range(len(batch_users)):
            gains = hits[i].float() / torch.log2(torch.arange(2, k + 2, device=device).float())
            ndcg += gains.sum().item()

    n_users = len(users)
    return precision / n_users, recall / n_users, ndcg / n_users

# ===================== 训练与评估 =====================
EPOCHS = 100
batch_size = 8192

results = {}

history = {"MLP": {"loss": [], "precision": [], "recall": [], "ndcg": []},
           "LightGCN": {"loss": [], "precision": [], "recall": [], "ndcg": []}}

# ----- 模型 1：MLP -----
mlp_model = TwoTowerModel(num_users, num_movies).to(device)
mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
print("\n Training TwoTower MLP Model...")
for epoch in range(1, EPOCHS + 1):
    loss = train_mlp(mlp_model, mlp_optimizer, train_df, num_movies, batch_size)
    pre, rec, ndcg = test(val_df, train_df, mlp_model.get_user_embedding(), mlp_model.get_movie_embedding())

    # 记录训练过程
    history["MLP"]["loss"].append(loss)
    history["MLP"]["precision"].append(pre)
    history["MLP"]["recall"].append(rec)
    history["MLP"]["ndcg"].append(ndcg)

    print(f"[MLP] Epoch {epoch:03d} | Loss={loss:.4f} | P@10={pre:.4f} | R@10={rec:.4f} | NDCG@10={ndcg:.4f}")

results["MLP"] = (pre, rec, ndcg)

# ----- 模型 2：LightGCN -----
lgn_model = LightGCNModel(num_users, num_movies).to(device)
lgn_optimizer = torch.optim.Adam(lgn_model.parameters(), lr=1e-3)
print("\n Training LightGCN Model...")
for epoch in range(1, EPOCHS + 1):
    loss = train_lightgcn(lgn_model, lgn_optimizer, train_df, train_edge_index, num_users, num_movies, batch_size)
    user_emb = lgn_model.get_user_embedding(train_edge_index)
    movie_emb = lgn_model.get_movie_embedding(train_edge_index)
    pre, rec, ndcg = test(val_df, train_df, user_emb, movie_emb)

    # 记录训练过程
    history["LightGCN"]["loss"].append(loss)
    history["LightGCN"]["precision"].append(pre)
    history["LightGCN"]["recall"].append(rec)
    history["LightGCN"]["ndcg"].append(ndcg)

    print(f"[LGN] Epoch {epoch:03d} | Loss={loss:.4f} | P@10={pre:.4f} | R@10={rec:.4f} | NDCG@10={ndcg:.4f}")

results["LightGCN"] = (pre, rec, ndcg)

# ===================== 输出最终对比结果 =====================
print("\n================ 最终评估结果对比 ================")
for name, (p, r, n) in results.items():
    print(f"{name:10s} | Precision@10={p:.4f} | Recall@10={r:.4f} | NDCG@10={n:.4f}")
print("===================================================")

# ===================== 可视化训练曲线 =====================
plt.figure(figsize=(14, 10))

# ---- Loss 曲线 ----
plt.subplot(2, 2, 1)
plt.plot(history["MLP"]["loss"], label='MLP')
plt.plot(history["LightGCN"]["loss"], label='LightGCN')
plt.title("Training Loss (BPR)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# ---- Precision ----
plt.subplot(2, 2, 2)
plt.plot(history["MLP"]["precision"], label='MLP')
plt.plot(history["LightGCN"]["precision"], label='LightGCN')
plt.title("Precision@10")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.legend()

# ---- Recall ----
plt.subplot(2, 2, 3)
plt.plot(history["MLP"]["recall"], label='MLP')
plt.plot(history["LightGCN"]["recall"], label='LightGCN')
plt.title("Recall@10")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()

# ---- NDCG ----
plt.subplot(2, 2, 4)
plt.plot(history["MLP"]["ndcg"], label='MLP')
plt.plot(history["LightGCN"]["ndcg"], label='LightGCN')
plt.title("NDCG@10")
plt.xlabel("Epoch")
plt.ylabel("NDCG")
plt.legend()

plt.tight_layout()
plt.savefig("MLPvsLGN_twotower_metrics.png", dpi=300, bbox_inches='tight')
plt.show()