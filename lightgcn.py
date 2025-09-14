import os.path as osp  

import torch  
from tqdm import tqdm  

# 导入PyG库
from torch_geometric.datasets import AmazonBook  
from torch_geometric.nn import LightGCN  
from torch_geometric.utils import degree  

import matplotlib.pyplot as plt  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Amazon')  
dataset = AmazonBook(path)  # 加载AmazonBook数据集
data = dataset[0]  # 获取数据集中的第一个图数据
num_users, num_books = data['user'].num_nodes, data['book'].num_nodes  # 获取用户和图书节点的数量
data = data.to_homogeneous().to(device)  # 将异构图转换为同构图并移动到指定设备

train_losses = []  
precisions = []  
recalls = []  

# 使用所有消息传递边作为训练标签：
batch_size = 8192  # 设置批处理大小
mask = data.edge_index[0] < data.edge_index[1]  # 创建掩码，去除重复边，只保留i<j的边
train_edge_label_index = data.edge_index[:, mask]  # 应用掩码获取训练边标签索引
train_loader = torch.utils.data.DataLoader(  # 数据加载
    range(train_edge_label_index.size(1)),  # 使用训练边索引范围作为数据集
    shuffle=True,  # 随机打乱
    batch_size=batch_size,  
)

model = LightGCN(  # 初始化LightGCN模型
    num_nodes=data.num_nodes,  # 设置节点总数
    embedding_dim=64,  # 设置嵌入维度
    num_layers=2,  # 设置网络层数
).to(device)  # 将模型移动到指定设备
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 初始化Adam优化器


def train():  # 定义训练函数
    total_loss = total_examples = 0  # 初始化总损失和总样本数

    for index in tqdm(train_loader):  # 使用进度条遍历训练数据
        # 采样正负标签：
        pos_edge_label_index = train_edge_label_index[:, index]  # 获取正样本边索引
        neg_edge_label_index = torch.stack([  # 生成负样本边索引（随机采样）
            pos_edge_label_index[0],  # 保持源节点不变
            torch.randint(num_users, num_users + num_books,  # 在图书节点中随机采样目标节点
                          (index.numel(), ), device=device)
        ], dim=0)
        edge_label_index = torch.cat([  # 拼接正负样本边索引
            pos_edge_label_index,
            neg_edge_label_index,
        ], dim=1)

        optimizer.zero_grad()  # 清零梯度
        pos_rank, neg_rank = model(data.edge_index, edge_label_index).chunk(2)  # 前向传播并分割正负样本得分

        loss = model.recommendation_loss(  # 计算推荐损失
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),  # 获取涉及的所有唯一节点
        )
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        total_loss += float(loss) * pos_rank.numel()  # 累加损失
        total_examples += pos_rank.numel()  # 累加样本数

    return total_loss / total_examples  # 返回平均损失


@torch.no_grad()  # 禁用梯度计算装饰器
def test(k: int):  # 定义测试函数，k为Top-K指标参数
    emb = model.get_embedding(data.edge_index)  # 获取节点嵌入
    user_emb, book_emb = emb[:num_users], emb[num_users:]  # 分割用户和图书嵌入

    precision = recall = total_examples = 0  # 初始化精确率、召回率和总样本数
    for start in range(0, num_users, batch_size):  # 按批处理遍历所有用户
        end = start + batch_size
        logits = user_emb[start:end] @ book_emb.t()  # 计算用户-图书相似度矩阵

        # 排除训练边（避免数据泄露）：
        mask = ((train_edge_label_index[0] >= start) &
                (train_edge_label_index[0] < end))
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - num_users] = float('-inf')

        # 计算精确率和召回率：
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)  # 初始化真实标签矩阵
        mask = ((data.edge_label_index[0] >= start) &
                (data.edge_label_index[0] < end))
        ground_truth[data.edge_label_index[0, mask] - start,
                     data.edge_label_index[1, mask] - num_users] = True  # 标记真实边
        node_count = degree(data.edge_label_index[0, mask] - start,  # 计算每个用户的真实边数
                            num_nodes=logits.size(0))

        topk_index = logits.topk(k, dim=-1).indices  # 获取Top-K预测索引
        isin_mat = ground_truth.gather(1, topk_index)  # 检查预测是否在真实边中

        precision += float((isin_mat.sum(dim=-1) / k).sum())  # 累加精确率
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())  # 累加召回率
        total_examples += int((node_count > 0).sum())  # 累加有效用户数

    return precision / total_examples, recall / total_examples  # 返回平均精确率和召回率

# 训练
for epoch in range(1, 101):  
    loss = train()  
    precision, recall = test(k=20)  # 在测试集上计算Precision@20和Recall@20
    train_losses.append(loss)  
    precisions.append(precision)  
    recalls.append(recall)  
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@20: '
          f'{precision:.4f}, Recall@20: {recall:.4f}')  

# 绘制训练曲线
plt.figure(figsize=(12, 4))  
plt.subplot(1, 3, 1)  
plt.plot(train_losses, label='Train Loss')  
plt.xlabel('Epoch')  
plt.ylabel('Loss')  
plt.title('Training Loss')  
plt.legend()  

plt.subplot(1, 3, 2)  
plt.plot(precisions, label='Precision@20')  
plt.xlabel('Epoch')  
plt.ylabel('Precision')  
plt.title('Precision@20')  
plt.legend()  

plt.subplot(1, 3, 3)  
plt.plot(recalls, label='Recall@20')  
plt.xlabel('Epoch')  
plt.ylabel('Recall')  
plt.title('Recall@20')  
plt.legend()  

plt.tight_layout()  
plt.savefig("training_metrics.png")  
plt.close()  