import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch_geometric.datasets import AmazonBook
from torch_geometric.nn import LightGCN, GCNConv
from torch_geometric.utils import degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== 数据加载 =====================
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Amazon')
dataset = AmazonBook(path)
data = dataset[0]
num_users, num_books = data['user'].num_nodes, data['book'].num_nodes
data = data.to_homogeneous().to(device)

batch_size = 8192
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
def train_one_epoch(model, optimizer, train_loader, edge_index, train_edge_label_index):
    total_loss = total_examples = 0
    model.train()

    for index in tqdm(train_loader, leave=False):
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_users, num_users + num_books,
                          (index.numel(), ), device=device)
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
def test(model, edge_index, k: int = 20):
    model.eval()
    emb = model.get_embedding(edge_index)
    user_emb, book_emb = emb[:num_users], emb[num_users:]

    precision = recall = total_examples = 0
    for start in range(0, num_users, batch_size):
        end = start + batch_size
        logits = user_emb[start:end] @ book_emb.t()

        # 屏蔽训练边
        mask = ((train_edge_label_index[0] >= start) & (train_edge_label_index[0] < end))
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - num_users] = float('-inf')

        # ground truth
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((data.edge_label_index[0] >= start) & (data.edge_label_index[0] < end))
        ground_truth[data.edge_label_index[0, mask] - start,
                     data.edge_label_index[1, mask] - num_users] = True
        node_count = degree(data.edge_label_index[0, mask] - start,
                            num_nodes=logits.size(0))

        topk_index = logits.topk(k, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)

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
    losses, precisions, recalls = [], [], []

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, optimizer, train_loader, data.edge_index, train_edge_label_index)
        precision, recall = test(model, data.edge_index, k=20)
        losses.append(loss)
        precisions.append(precision)
        recalls.append(recall)
        print(f"[{name}] Epoch {epoch:03d} | Loss: {loss:.4f} | P@20: {precision:.4f} | R@20: {recall:.4f}")

    return losses, precisions, recalls


# ===================== 主流程 =====================
epochs = 100
lgn_loss, lgn_p, lgn_r = run_experiment(LightGCN, "LightGCN", epochs=epochs)
gcn_loss, gcn_p, gcn_r = run_experiment(GCNRecommender, "GCN", epochs=epochs)

# ===================== 可视化对比 =====================
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(lgn_loss, label='LightGCN')
plt.plot(gcn_loss, label='GCN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(lgn_p, label='LightGCN')
plt.plot(gcn_p, label='GCN')
plt.xlabel('Epoch')
plt.ylabel('Precision@20')
plt.title('Precision@20')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(lgn_r, label='LightGCN')
plt.plot(gcn_r, label='GCN')
plt.xlabel('Epoch')
plt.ylabel('Recall@20')
plt.title('Recall@20')
plt.legend()

plt.tight_layout()
plt.savefig("lgn_vs_gcn.png")
plt.close()
