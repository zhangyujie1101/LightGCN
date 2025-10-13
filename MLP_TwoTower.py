# ===================== 导入库 =====================
import os.path as osp
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

# ===================== 双塔模型定义 =====================
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64, hidden_dim=128):
        super().__init__()
        # 用户 & 物品嵌入
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)

        # 用户塔（MLP）
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # 物品塔（MLP）
        self.movie_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, user_idx, movie_idx):
        user_vec = self.user_mlp(self.user_emb(user_idx))
        movie_vec = self.movie_mlp(self.movie_emb(movie_idx))
        score = (user_vec * movie_vec).sum(dim=-1)
        return score

    def get_user_embedding(self):
        return self.user_mlp(self.user_emb.weight)

    def get_movie_embedding(self):
        return self.movie_mlp(self.movie_emb.weight)

# ===================== 训练准备 =====================
model = TwoTowerModel(num_users, num_movies).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size = 8192

train_loader = torch.utils.data.DataLoader(
    range(len(train_df)), shuffle=True, batch_size=batch_size
)

# ===================== 训练函数（BCE损失） =====================
def train():
    model.train()
    total_loss = total_examples = 0

    for idx in tqdm(train_loader):
        batch = train_df.iloc[idx]
        user_idx = torch.tensor(batch['user_idx'].values, dtype=torch.long, device=device)
        movie_idx = torch.tensor(batch['movie_idx'].values, dtype=torch.long, device=device)

        # 正样本 = 1，负样本随机采样
        neg_movie_idx = torch.randint(0, num_movies, (len(idx),), device=device)
        neg_user_idx = user_idx

        optimizer.zero_grad()

        pos_score = model(user_idx, movie_idx)
        neg_score = model(neg_user_idx, neg_movie_idx)

        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
        
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * len(idx)
        total_examples += len(idx)

    return total_loss / total_examples

# ===================== 测试指标（Recall@10, Precision@10, NDCG@10） =====================
@torch.no_grad()
def test(eval_df, k=10):
    model.eval()
    user_emb = model.get_user_embedding()
    movie_emb = model.get_movie_embedding()

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
        ground_truth = torch.zeros((len(batch_users), num_movies), dtype=torch.bool, device=device)
        for i, u in enumerate(batch_users):
            gt_movies = eval_df[eval_df['user_idx'] == u.item()]['movie_idx'].values
            ground_truth[i, gt_movies] = True

        topk = logits.topk(k, dim=-1).indices
        hits = ground_truth.gather(1, topk)

        # Precision, Recall
        precision += (hits.sum(dim=-1) / k).sum().item()
        recall += (hits.sum(dim=-1) / ground_truth.sum(dim=-1).clamp(1e-6)).sum().item()

        # NDCG
        for i in range(len(batch_users)):
            gains = hits[i].float() / torch.log2(torch.arange(2, k + 2, device=device).float())
            ndcg += gains.sum().item()

    n_users = len(users)
    return precision / n_users, recall / n_users, ndcg / n_users

# ===================== 训练循环 =====================
train_losses, val_precs, val_recs, val_ndcgs = [], [], [], []

for epoch in range(1, 101):
    loss = train()
    pre, rec, ndcg = test(val_df, k=10)
    train_losses.append(loss)
    val_precs.append(pre)
    val_recs.append(rec)
    val_ndcgs.append(ndcg)

    print(f"Epoch {epoch:03d}: Loss={loss:.4f}, Precision@10={pre:.4f}, Recall@10={rec:.4f}, NDCG@10={ndcg:.4f}")

# ===================== 绘图 =====================
plt.figure(figsize=(15,4))
plt.subplot(1,4,1)
plt.plot(train_losses); plt.title("Training Loss")
plt.subplot(1,4,2)
plt.plot(val_precs, label="Precision@10", color='orange'); plt.legend()
plt.subplot(1,4,3)
plt.plot(val_recs, label="Recall@10", color='green'); plt.legend()
plt.subplot(1,4,4)
plt.plot(val_ndcgs, label="NDCG@10", color='red'); plt.legend()
plt.tight_layout()
plt.savefig("twotower_metrics.png", dpi=300, bbox_inches='tight')
plt.close()
