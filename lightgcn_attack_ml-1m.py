# -*- coding: utf-8 -*-
"""
Movielens Poisoning Attack Experiments
启发式攻击: RandomAttack, AverageAttack, AoPAttack, BandwagonAttack
生成式近似攻击: RAPU (基于 item-embedding 相似度合成评分)
只在用户端注入假用户（不修改项目/movie 原始数据）
"""

import os.path as osp
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import LightGCN

# ============ 配置 ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = './datasets/ml-1m/'
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 攻击配置
attack_user_fraction = 0.01      # 注入假用户占原始用户的比例
filler_per_fake = 50             # 每个假用户标注多少个 filler items
target_item = None               # 若想攻击特定 movie_idx（映射后的索引），可填 int，否则为 None（针对推动热门/随机）
topk = 20                        # 评估Top-K（Precision@K / Recall@K）
num_epochs = 60                  # 每次训练 epoch 数
embedding_dim = 64
batch_size = 8192
lr = 0.001

# ===================== 加载 MovieLens-1M 数据集 =====================
def load_data(data_path):
    ratings = pd.read_csv(osp.join(data_path, 'ratings.dat'),
                          sep='::',
                          engine='python',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
    movies = pd.read_csv(osp.join(data_path, 'movies.dat'),
                         sep='::',
                         engine='python',
                         names=['movie_id', 'title', 'genres'],
                         encoding='latin-1')
    users = pd.read_csv(osp.join(data_path, 'users.dat'),
                        sep='::',
                        engine='python',
                        names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
    print(f"数据集统计: {len(ratings)} 条评分, {len(users)} 个用户, {len(movies)} 部电影")

    # 重新映射用户和电影ID为连续整数（从0开始）
    user_mapping = {orig: new for new, orig in enumerate(ratings['user_id'].unique())}
    movie_mapping = {orig: new for new, orig in enumerate(ratings['movie_id'].unique())}
    ratings['user_idx'] = ratings['user_id'].map(user_mapping)
    ratings['movie_idx'] = ratings['movie_id'].map(movie_mapping)

    num_users = len(user_mapping)
    num_movies = len(movie_mapping)
    print(f"重新映射后: {num_users} 个用户, {num_movies} 部电影")
    return ratings, num_users, num_movies, movies

ratings, orig_num_users, num_movies, movies_df = load_data(data_path)

# ===================== 数据切分（8:1:1） =====================
train_df, temp_df = train_test_split(ratings, test_size=0.2, random_state=seed)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

print(f"训练集: {len(train_df)} 条评分, 验证: {len(val_df)}, 测试: {len(test_df)}")

# ============ 辅助函数：构建图边索引 ============
def build_edge_index_from_df(df, num_users, num_movies):
    user_nodes = torch.tensor(df['user_idx'].values, dtype=torch.long)
    movie_nodes = torch.tensor(df['movie_idx'].values, dtype=torch.long) + num_users
    edge_index = torch.stack([torch.cat([user_nodes, movie_nodes]),
                              torch.cat([movie_nodes, user_nodes])], dim=0)
    return edge_index

# ============ 模型训练 / 评估 工具函数 ============
def make_data_object(train_df, num_users, num_movies):
    edge_index = build_edge_index_from_df(train_df, num_users, num_movies)
    data_obj = Data(edge_index=edge_index, num_nodes=num_users + num_movies).to(device)
    return data_obj

def prepare_train_loader(data_obj, num_users):
    # 去重正样本（因为双向边）
    mask = data_obj.edge_index[0] < data_obj.edge_index[1]
    train_edge_label_index = data_obj.edge_index[:, mask]
    idx_range = range(train_edge_label_index.size(1))
    loader = torch.utils.data.DataLoader(idx_range, shuffle=True, batch_size=batch_size)
    # 返回正样本索引用于训练和测试时过滤
    return train_edge_label_index.to(device), loader

def init_model_and_optimizer(num_nodes):
    model = LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt

# BPR style training
def run_train(model, optimizer, data_obj, train_edge_label_index, train_loader, num_users):
    model.train()
    total_loss = total_examples = 0.0
    for index in tqdm(train_loader, desc="Training", leave=False):
        pos_edge = train_edge_label_index[:, index].to(device)
        # 随机负采样物品
        neg_items = torch.randint(num_users, num_users + num_movies, (pos_edge.size(1),), device=device)
        neg_edge = torch.stack([pos_edge[0], neg_items], dim=0)
        edge_label_index = torch.cat([pos_edge, neg_edge], dim=1)
        optimizer.zero_grad()
        pos_rank, neg_rank = model(data_obj.edge_index, edge_label_index).chunk(2)
        loss = model.recommendation_loss(pos_rank, neg_rank, node_id=edge_label_index.unique())
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()
    return total_loss / (total_examples + 1e-12)

@torch.no_grad()
def eval_model(model, data_obj, train_edge_label_index, eval_edge_label_index, num_users, k=20):
    model.eval()
    emb = model.get_embedding(data_obj.edge_index)
    user_emb = emb[:num_users]
    item_emb = emb[num_users:num_users + num_movies]
    precision = recall = total_users_with_gt = 0.0

    test_users = eval_edge_label_index[0, eval_edge_label_index[0] < num_users].unique()
    test_users = test_users.tolist()

    # 将 train interactions per user 汇总到字典，用于屏蔽训练中已交互的电影
    train_user_items = {}
    mask = train_edge_label_index[0] < num_users
    for i in range(train_edge_label_index.size(1)):
        u = int(train_edge_label_index[0, i].item())
        v = int(train_edge_label_index[1, i].item() - num_users)
        train_user_items.setdefault(u, set()).add(v)

    for start in range(0, len(test_users), batch_size):
        end = min(start + batch_size, len(test_users))
        u_batch = torch.tensor(test_users[start:end], device=device, dtype=torch.long)
        logits = user_emb[u_batch] @ item_emb.t()

        # 屏蔽该用户训练集里的物品
        for i, u in enumerate(u_batch):
            u_int = int(u.item())
            if u_int in train_user_items and len(train_user_items[u_int]) > 0:
                logits[i, list(train_user_items[u_int])] = float('-inf')

        # 构建 ground truth 矩阵
        ground_truth = torch.zeros((len(u_batch), num_movies), dtype=torch.bool, device=device)
        for i, u in enumerate(u_batch):
            mask = (eval_edge_label_index[0] == u) & (eval_edge_label_index[1] >= num_users)
            true_items = (eval_edge_label_index[1, mask] - num_users).long()
            if true_items.numel() > 0:
                ground_truth[i, true_items] = True

        node_count = ground_truth.sum(dim=1)
        topk_idx = logits.topk(k, dim=-1).indices
        isin = ground_truth.gather(1, topk_idx)
        precision += float((isin.sum(dim=-1) / k).sum())
        recall += float((isin.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_users_with_gt += int((node_count > 0).sum().item())

    if total_users_with_gt == 0:
        return 0.0, 0.0
    return precision / total_users_with_gt, recall / total_users_with_gt

# ============ 攻击策略实现（注入假用户，仅修改 train_df） ============
def RandomAttack(train_df, num_users, num_movies, num_fake_users, filler_size, rating_value=5.0, target_item=None):
    """
    为每个假用户随机选择 filler_size 个物品并赋固定高评分（rating_value）；
    如果提供 target_item，会保证 target_item 出现在部分假用户的评分中（以增加攻击效果）。
    返回新的 train_df, 新的 num_users（扩展后）·
    """
    fake_rows = []
    start_idx = num_users
    item_pool = list(range(num_movies))
    for fu in range(num_fake_users):
        u_idx = start_idx + fu
        # 随机选取 filler items
        items = random.sample(item_pool, k=min(filler_size, num_movies))
        # 若指定 target_item，确保一部分假用户包含它
        if target_item is not None and fu % 2 == 0 and target_item not in items:
            # 替换第一个元素
            items[0] = target_item
        for it in items:
            fake_rows.append({'user_idx': u_idx, 'movie_idx': int(it), 'rating': rating_value})
    fake_df = pd.DataFrame(fake_rows)
    new_train = pd.concat([train_df, fake_df], ignore_index=True)
    new_num_users = num_users + num_fake_users
    return new_train, new_num_users

def AverageAttack(train_df, num_users, num_movies, num_fake_users, filler_size, rating_value=5.0, target_item=None):
    """
    对每个 item 计算均值评分，假用户对随机选的 items 赋予 item 的均值或调高的均值
    """
    avg_rating = train_df.groupby('movie_idx')['rating'].mean().to_dict()
    fake_rows = []
    start_idx = num_users
    item_pool = list(range(num_movies))
    for fu in range(num_fake_users):
        u_idx = start_idx + fu
        items = random.sample(item_pool, k=min(filler_size, num_movies))
        if target_item is not None and fu % 2 == 0 and target_item not in items:
            items[0] = target_item
        for it in items:
            r = avg_rating.get(it, 3.0)
            # 将均值向上推到 rating_value（但不超过 rating_value）
            r = min(r + 1.0, rating_value)
            fake_rows.append({'user_idx': u_idx, 'movie_idx': int(it), 'rating': r})
    fake_df = pd.DataFrame(fake_rows)
    new_train = pd.concat([train_df, fake_df], ignore_index=True)
    return new_train, num_users + num_fake_users

def AoPAttack(train_df, num_users, num_movies, num_fake_users, filler_size, rating_value=5.0, target_item=None):
    """
    AoP (Average of Popularity) attack: 假用户优先给流行度高的物品高分（filler选择热门物品）
    """
    pop = train_df['movie_idx'].value_counts().to_dict()
    # 将 items 按流行度排序并取前一部分作为热门池
    popular_items = [int(x) for x, _ in sorted(pop.items(), key=lambda kv: -kv[1])]
    if len(popular_items) == 0:
        popular_items = list(range(num_movies))
    fake_rows = []
    start_idx = num_users
    pop_pool = popular_items[:max(100, int(0.1 * num_movies))]  # 取前 10% 或至少 100
    for fu in range(num_fake_users):
        u_idx = start_idx + fu
        # 选取部分热门物品作为 filler（尽量选热门）
        items = random.sample(pop_pool, k=min(filler_size, len(pop_pool)))
        if target_item is not None and target_item not in items and fu % 2 == 0:
            items[0] = target_item
        for it in items:
            fake_rows.append({'user_idx': u_idx, 'movie_idx': int(it), 'rating': rating_value})
    fake_df = pd.DataFrame(fake_rows)
    new_train = pd.concat([train_df, fake_df], ignore_index=True)

    print(f"[{attack_type}] 注入假用户数: {num_fake_users}, 训练集总数: {len(new_train)}")
    print(new_train.tail(10))

    return new_train, num_users + num_fake_users

def BandwagonAttack(train_df, num_users, num_movies, num_fake_users, filler_size, rating_value=5.0, target_item=None):
    """
    Bandwagon：部分填充热门物品（吸引模型），并在每个假用户中加入 target_item（若给定），其他 position 随机
    """
    pop = train_df['movie_idx'].value_counts().to_dict()
    popular_items = [int(x) for x, _ in sorted(pop.items(), key=lambda kv: -kv[1])]
    if len(popular_items) == 0:
        popular_items = list(range(num_movies))
    pop_pool = popular_items[:max(50, int(0.05 * num_movies))]
    fake_rows = []
    start_idx = num_users
    for fu in range(num_fake_users):
        u_idx = start_idx + fu
        items = []
        # 保证部分热门
        num_pop_fill = max(1, int(0.2 * filler_size))
        if len(pop_pool) >= num_pop_fill:
            items += random.sample(pop_pool, k=num_pop_fill)
        # 其余随机选
        remaining = max(0, filler_size - len(items))
        other_pool = [i for i in range(num_movies) if i not in items]
        if remaining > 0 and len(other_pool) > 0:
            items += random.sample(other_pool, k=min(remaining, len(other_pool)))
        # 插入 target_item
        if target_item is not None and target_item not in items:
            items[0] = target_item
        for it in items:
            fake_rows.append({'user_idx': u_idx, 'movie_idx': int(it), 'rating': rating_value})
    fake_df = pd.DataFrame(fake_rows)
    new_train = pd.concat([train_df, fake_df], ignore_index=True)
    return new_train, num_users + num_fake_users

def RAPU_Attack(train_df, num_users, num_movies, num_fake_users, filler_size, model_for_gen, data_obj, num_epochs_inner=3, rating_value=5.0, target_item=None):
    """
    RAPU 生成式简化近似实现（实用版）:
    - 使用给定的 model_for_gen（LightGCN 已训练或初步训练过）来提取 item embedding
    - 对于每个假用户，如果提供 target_item，则优先把 target_item 标成高分
    - 其余 filler 选取与 target_item 相似（embedding 相似）或总体热门/相似的 item，并打高分
    说明：这不是学术上严格的 RAPU（原论文用生成模型），但能模拟“生成式高质量伪造评分”的效果
    """
    # 获取 item embeddings
    model_for_gen.eval()
    with torch.no_grad():
        emb = model_for_gen.get_embedding(data_obj.edge_index).cpu()
    item_emb = emb[num_users:num_users + num_movies].numpy()  # shape (num_movies, dim)

    # 计算 pairwise 相似度（cos）
    norm = np.linalg.norm(item_emb, axis=1, keepdims=True) + 1e-9
    item_emb_norm = item_emb / norm
    sim_matrix = item_emb_norm @ item_emb_norm.T  # num_movies x num_movies

    fake_rows = []
    start_idx = num_users
    all_items = list(range(num_movies))
    for fu in range(num_fake_users):
        u_idx = start_idx + fu
        items = []
        if target_item is not None:
            items.append(target_item)
            # 选相似性最高的若干 items（不包含自己）
            sim_scores = sim_matrix[target_item]
            top_sim_idx = np.argsort(-sim_scores)
            cnt = min(filler_size - 1, len(top_sim_idx) - 1)
            i = 0
            while len(items) < filler_size and i < len(top_sim_idx):
                cand = int(top_sim_idx[i])
                if cand != target_item and cand not in items:
                    items.append(cand)
                i += 1
        else:
            # 若无显式 target_item，则为假用户从热门+相似中心采样一些高相似群组
            # 选一个中心 item（热门或随机）
            pop = train_df['movie_idx'].value_counts()
            center = int(pop.idxmax()) if not pop.empty else random.choice(all_items)
            items.append(center)
            sim_scores = sim_matrix[center]
            top_sim_idx = np.argsort(-sim_scores)
            i = 0
            while len(items) < filler_size and i < len(top_sim_idx):
                cand = int(top_sim_idx[i])
                if cand != center and cand not in items:
                    items.append(cand)
                i += 1
        for it in items:
            fake_rows.append({'user_idx': u_idx, 'movie_idx': int(it), 'rating': rating_value})
    fake_df = pd.DataFrame(fake_rows)
    new_train = pd.concat([train_df, fake_df], ignore_index=True)
    return new_train, num_users + num_fake_users

# ============ 实验主循环：基线 -> 各种攻击 -> 训练评估 ============
def train_and_evaluate(train_df_in, val_df, test_df, num_users_in, num_movies, epochs=num_epochs, do_print=True):
    # 构建数据对象
    data_obj = make_data_object(train_df_in, num_users_in, num_movies)
    # val/test edge label index
    val_edge_label_index = build_edge_index_from_df(val_df, num_users_in, num_movies).to(device)
    test_edge_label_index = build_edge_index_from_df(test_df, num_users_in, num_movies).to(device)

    # 训练用正样本索引与 loader
    train_edge_label_index, train_loader = prepare_train_loader(data_obj, num_users_in)
    model, opt = init_model_and_optimizer(data_obj.num_nodes)

    # 训练循环
    for epoch in range(1, epochs + 1):
        loss = run_train(model, opt, data_obj, train_edge_label_index, train_loader, num_users_in)
        if epoch % 10 == 0 and do_print:
            val_prec, val_rec = eval_model(model, data_obj, train_edge_label_index, val_edge_label_index, num_users_in, k=topk)
            test_prec, test_rec = eval_model(model, data_obj, train_edge_label_index, test_edge_label_index, num_users_in, k=topk)
            print(f"Epoch {epoch}/{epochs}  Loss {loss:.4f}  Val P@{topk}:{val_prec:.4f} R@{topk}:{val_rec:.4f}  Test P@{topk}:{test_prec:.4f} R@{topk}:{test_rec:.4f}")
    # 最终评估
    val_prec, val_rec = eval_model(model, data_obj, train_edge_label_index, val_edge_label_index, num_users_in, k=topk)
    test_prec, test_rec = eval_model(model, data_obj, train_edge_label_index, test_edge_label_index, num_users_in, k=topk)
    if do_print:
        print(f"FINAL  Val P@{topk}:{val_prec:.4f}  R@{topk}:{val_rec:.4f}   Test P@{topk}:{test_prec:.4f}  R@{topk}:{test_rec:.4f}")
    return {
        'model': model,
        'data_obj': data_obj,
        'train_edge_label_index': train_edge_label_index,
        'val_prec': val_prec,
        'val_rec': val_rec,
        'test_prec': test_prec,
        'test_rec': test_rec,
        'num_users': num_users_in
    }



# ============ 攻击设置与执行 ============
attack_choices = ['RandomAttack', 'AverageAttack', 'AoPAttack', 'BandwagonAttack', 'RAPU']
attack_type = 'AoPAttack'

num_fake = max(1, int(orig_num_users * attack_user_fraction))
print(f"\n选择攻击形式: {attack_type}")
print(f"注入假用户数: {num_fake} (原始用户数 {orig_num_users})")



# ============ 运行基线训练 ============
print("=== 训练基线模型（无攻击） ===")
baseline_result = train_and_evaluate(train_df, val_df, test_df, orig_num_users, num_movies, epochs=num_epochs, do_print=True)

attack_results = {}

# 1) RandomAttack
if attack_type == 'RandomAttack':
    print("=== 执行 RandomAttack ===")
    train_mod, n_users_mod = RandomAttack(train_df, orig_num_users, num_movies, num_fake, filler_per_fake, rating_value=5.0, target_item=target_item)

# 2) AverageAttack
elif attack_type == 'AverageAttack':
    print("=== 执行 AverageAttack ===")
    train_mod, n_users_mod = AverageAttack(train_df, orig_num_users, num_movies, num_fake, filler_per_fake, rating_value=5.0, target_item=target_item)

# 3) AoPAttack
elif attack_type == 'AoPAttack':
    print("=== 执行 AoPAttack ===")
    train_mod, n_users_mod = AoPAttack(train_df, orig_num_users, num_movies, num_fake, filler_per_fake, rating_value=5.0, target_item=target_item)

# 4) BandwagonAttack
elif attack_type == 'BandwagonAttack':
    print("=== 执行 BandwagonAttack ===")
    train_mod, n_users_mod = BandwagonAttack(train_df, orig_num_users, num_movies, num_fake, filler_per_fake, rating_value=5.0, target_item=target_item)

# 5) RAPU (生成式近似) - 基于基线模型的 item embedding 生成假用户
elif attack_type == 'RAPU':
    print("=== 执行 RAPU (生成式近似) ===")
    gen_model = baseline_result['model']
    gen_data_obj = baseline_result['data_obj']
    train_mod, n_users_mod = RAPU_Attack(train_df, orig_num_users, num_movies, num_fake, filler_per_fake,
                                         model_for_gen=gen_model, data_obj=gen_data_obj,
                                         rating_value=5.0, target_item=target_item)
else:
    raise ValueError(f"未知攻击类型: {attack_type}")

attack_result = train_and_evaluate(train_mod, val_df, test_df, n_users_mod, num_movies, epochs=num_epochs, do_print=True)
print(f"原始用户数: {orig_num_users}, 新用户数: {n_users_mod}")

# ===================== 输出攻击结果 =====================
print("\n====================== 实验结果汇总 ======================")

print("\n--- 基线模型（无攻击） ---")
print(f"Val Precision@20: {baseline_result['val_prec']:.4f}")
print(f"Val Recall@20:    {baseline_result['val_rec']:.4f}")
print(f"Test Precision@20: {baseline_result['test_prec']:.4f}")
print(f"Test Recall@20:    {baseline_result['test_rec']:.4f}")

print(f"\n--- 攻击形式: {attack_type} ---")
print(f"Val Precision@20: {attack_result['val_prec']:.4f}")
print(f"Val Recall@20:    {attack_result['val_rec']:.4f}")
print(f"Test Precision@20: {attack_result['test_prec']:.4f}")
print(f"Test Recall@20:    {attack_result['test_rec']:.4f}")

print("\n==========================================================\n")
