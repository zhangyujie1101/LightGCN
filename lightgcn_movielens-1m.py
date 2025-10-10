# ===================== 导入库 =====================
import os.path as osp
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# 导入PyG库
from torch_geometric.data import Data
from torch_geometric.nn import LightGCN

import matplotlib.pyplot as plt

# ===================== 设置设备 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== 加载MovieLens-1M数据集 =====================
def load_data(data_path):
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
data_path = './datasets/ml-1m/'
ratings, num_users, num_movies = load_data(data_path)

# ===================== 数据划分 (8:1:1) =====================
# 划分训练集和临时测试集 (8:2)
train_df, temp_df = train_test_split(ratings, test_size=0.2, random_state=42)
# 将临时测试集划分为验证集和测试集 (1:1)
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

# ===================== 初始化列表 =====================
train_losses = []
val_precisions = []
val_recalls = []
test_precisions = []
test_recalls = []

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

# ===================== 初始化模型和优化器 =====================
model = LightGCN(
    num_nodes=data.num_nodes,
    embedding_dim=64,
    num_layers=2,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ===================== 训练函数 =====================
'''
# BPR损失
def train():
    total_loss = total_examples = 0

    for index in tqdm(train_loader):
        # 采样正样本边
        pos_edge_label_index = train_edge_label_index[:, index]

        # 生成负样本边
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_users, num_users + num_movies,
                          (index.numel(),), device=device)
        ], dim=0)

        edge_label_index = torch.cat([
            pos_edge_label_index,
            neg_edge_label_index,
        ], dim=1)

        optimizer.zero_grad()
        pos_rank, neg_rank = model(data.edge_index, edge_label_index).chunk(2)
        loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples
'''

'''
# MSE损失
def train():
    total_loss = total_examples = 0
    model.train()

    for index in tqdm(train_loader):
        # 采样正样本边
        pos_edge_label_index = train_edge_label_index[:, index]

        # 提取这些正样本对应的评分（注意此时我们有评分信息）
        ratings_batch = torch.tensor(
            train_df.iloc[index]['rating'].values, dtype=torch.float32, device=device
        )

        optimizer.zero_grad()

        # 获取节点embedding
        emb = model.get_embedding(data.edge_index)

        # 计算每个用户-物品对的预测分数（点积）
        pred_ratings = (emb[pos_edge_label_index[0]] * emb[pos_edge_label_index[1]]).sum(dim=-1)

        # === 计算L2范数损失（即MSE损失） ===
        loss = torch.mean((pred_ratings - ratings_batch) ** 2)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * len(index)
        total_examples += len(index)

    return total_loss / total_examples
'''


# BCE损失
def train():
    total_loss = total_examples = 0
    model.train()

    for index in tqdm(train_loader):
        # 采样正样本边（用户-物品对）
        pos_edge_label_index = train_edge_label_index[:, index]

        # 生成负样本边（用户-随机物品对）
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_users, num_users + num_movies,
                          (index.numel(),), device=device)
        ], dim=0)

        # 合并正负样本
        edge_label_index = torch.cat([
            pos_edge_label_index,
            neg_edge_label_index
        ], dim=1)

        # 创建标签：正样本=1，负样本=0
        edge_label = torch.cat([
            torch.ones(pos_edge_label_index.size(1), device=device),
            torch.zeros(neg_edge_label_index.size(1), device=device)
        ])

        optimizer.zero_grad()

        # 获取节点embedding
        emb = model.get_embedding(data.edge_index)

        # 计算预测得分（点积作为交互概率的近似）
        pred = (emb[edge_label_index[0]] * emb[edge_label_index[1]]).sum(dim=-1)

        # 使用Sigmoid激活，再计算二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * len(index)
        total_examples += len(index)

    return total_loss / total_examples

# ===================== 测试函数 =====================
@torch.no_grad()
def test(edge_label_index, k: int):
    """
    通用的测试函数，可以用于验证集和测试集
    """
    emb = model.get_embedding(data.edge_index)
    user_emb, movie_emb = emb[:num_users], emb[num_users:num_users + num_movies]

    precision = recall = total_examples = 0

    # 获取当前测试集中的所有用户
    test_users = edge_label_index[0, edge_label_index[0] < num_users].unique()

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
            mask = (edge_label_index[0] == user) & (edge_label_index[1] >= num_users)
            test_movies = edge_label_index[1, mask] - num_users
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


# ===================== 训练循环 =====================
for epoch in range(1, 101):
    loss = train()

    # 在验证集上测试
    val_precision, val_recall = test(data.val_edge_label_index, k=20)
    # 在测试集上测试（可选，通常只在最后测试）
    if epoch % 10 == 0:  # 每10个epoch在测试集上测试一次
        test_precision, test_recall = test(data.test_edge_label_index, k=20)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
    else:
        test_precision, test_recall = 0, 0

    # 记录指标
    train_losses.append(loss)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
          f'Val Precision@20: {val_precision:.4f}, Val Recall@20: {val_recall:.4f}'
          f'{f", Test Precision@20: {test_precision:.4f}, Test Recall@20: {test_recall:.4f}" if epoch % 10 == 0 else ""}')

# ===================== 绘图 =====================
plt.figure(figsize=(15, 4))

# 训练损失
plt.subplot(1, 4, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# 验证集 Precision@20
plt.subplot(1, 4, 2)
plt.plot(val_precisions, label='Val Precision@20', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Validation Precision@20')
plt.legend()

# 验证集 Recall@20
plt.subplot(1, 4, 3)
plt.plot(val_recalls, label='Val Recall@20', color='green')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Validation Recall@20')
plt.legend()

# 测试集指标（稀疏点）
plt.subplot(1, 4, 4)
test_epochs = list(range(10, 101, 10))
plt.plot(test_epochs, test_precisions, 'o-', label='Test Precision@20', color='red')
plt.plot(test_epochs, test_recalls, 's-', label='Test Recall@20', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Test Metrics')
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png", dpi=300, bbox_inches='tight')
plt.close()