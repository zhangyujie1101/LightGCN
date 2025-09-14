# ===================== 导入库 =====================
import os.path as osp 
import torch  
from tqdm import tqdm  

# 导入PyG库
from torch_geometric.datasets import AmazonBook  
from torch_geometric.nn import LightGCN  
from torch_geometric.utils import degree  

import matplotlib.pyplot as plt  


# ===================== 设置设备 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


# ===================== 加载数据集 =====================
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Amazon')  
dataset = AmazonBook(path)  
data = dataset[0]  # 获取数据集中的第一个图数据
num_users, num_books = data['user'].num_nodes, data['book'].num_nodes  # 获取用户和图书节点的数量
data = data.to_homogeneous().to(device)  # 将异构图转换为同构图


# ===================== 初始化列表 =====================
train_losses = []  
precisions = []  
recalls = []  


# ===================== 准备训练数据 =====================
# 使用所有消息传递边作为训练标签：
batch_size = 8192  # 设置批处理大小
mask = data.edge_index[0] < data.edge_index[1]  # 创建掩码，去除重复边，只保留i<j的边
train_edge_label_index = data.edge_index[:, mask]  # 应用掩码获取训练边标签索引
train_loader = torch.utils.data.DataLoader(  # 数据加载
    range(train_edge_label_index.size(1)),  # 使用训练边索引范围作为数据集
    shuffle=True,  # 随机打乱
    batch_size=batch_size,  
)


# ===================== 初始化模型和优化器 =====================
model = LightGCN(  
    num_nodes=data.num_nodes,  # 节点数
    embedding_dim=64,  # 嵌入维度
    num_layers=2,  # 网络层数
).to(device)

# Adam优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  


# ===================== 训练函数 =====================
def train():  
    """
    训练函数：执行一个epoch的训练。
    步骤：
      1. 从训练边中采样一个批次的正样本边。
      2. 为每个正样本边生成一个负样本边（随机采样未交互的图书）。
      3. 计算模型输出（正负样本的得分）。
      4. 计算推荐损失（基于BPR损失）。
      5. 反向传播和优化。
    返回平均训练损失。
    """
    total_loss = total_examples = 0  # 初始化总损失和总样本数

    for index in tqdm(train_loader):  
        # 采样正样本边：从训练边中获取当前批次的边索引
        pos_edge_label_index = train_edge_label_index[:, index]  
        # 生成负样本边：源节点与正样本相同，目标节点随机从图书节点中采样（模拟未交互的图书）
        neg_edge_label_index = torch.stack([  
            pos_edge_label_index[0],  # 源节点（用户）
            torch.randint(num_users, num_users + num_books,  # 目标节点：随机图书索引（范围从num_users到num_users+num_books）
                          (index.numel(), ), device=device)
        ], dim=0)
        # 合并正负样本边索引：第一半是正样本，第二半是负样本
        edge_label_index = torch.cat([  
            pos_edge_label_index,
            neg_edge_label_index,
        ], dim=1)

        optimizer.zero_grad()  # 清零梯度：防止梯度累积
        # 模型前向传播：输入整个图的边索引和当前批次的边标签索引，得到正负样本的得分
        # model返回的得分张量中，前一半是正样本得分，后一半是负样本得分
        pos_rank, neg_rank = model(data.edge_index, edge_label_index).chunk(2)  
        # 计算推荐损失：使用LightGCN内置的BPR损失函数，比较正样本和负样本的得分
        loss = model.recommendation_loss(  
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),  # 获取涉及的所有唯一节点（用于嵌入查找）
        )
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        total_loss += float(loss) * pos_rank.numel()  # 累加损失（损失是平均值，乘以样本数得到总损失）
        total_examples += pos_rank.numel()  # 累加样本数（正样本数量）

    return total_loss / total_examples  # 返回平均损失


# ===================== 测试函数 =====================
@torch.no_grad()  # 禁用梯度计算：节省内存和计算资源，用于评估
def test(k: int):  
    """
    测试函数：评估模型性能，计算Precision@K和Recall@K。
    步骤：
      1. 获取所有节点的嵌入。
      2. 分割嵌入为用户嵌入和图书嵌入。
      3. 按批次处理用户，计算用户-图书相似度矩阵。
      4. 排除训练边（避免数据泄露）。
      5. 基于测试边计算真实标签。
      6. 计算Top-K预测，并比较真实标签得到Precision和Recall。
    返回平均Precision@K和平均Recall@K。
    """
    emb = model.get_embedding(data.edge_index)  # 通过模型获取所有节点的嵌入（使用整个图结构）
    user_emb, book_emb = emb[:num_users], emb[num_users:]  # 分割嵌入：前num_users个是用户嵌入，其余是图书嵌入

    precision = recall = total_examples = 0  # 初始化精确率、召回率和有效用户数
    for start in range(0, num_users, batch_size):  # 按批处理遍历所有用户
        end = start + batch_size
        # 计算当前批次用户与所有图书的相似度矩阵（内积）
        logits = user_emb[start:end] @ book_emb.t() 

        # 排除训练边：将训练边对应的相似度设置为负无穷，防止模型预测已见过的边
        mask = ((train_edge_label_index[0] >= start) &
                (train_edge_label_index[0] < end))
        # 调整索引：将全局用户索引转换为当前批次内的局部索引
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - num_users] = float('-inf')

        # 创建真实标签矩阵：标记测试边（假设data.edge_label_index存在，表示测试边）
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)  
        mask = ((data.edge_label_index[0] >= start) &
                (data.edge_label_index[0] < end))
        # 标记测试边：将测试边对应的位置设为True
        ground_truth[data.edge_label_index[0, mask] - start,
                     data.edge_label_index[1, mask] - num_users] = True  
        # 计算每个用户的测试边数量（度）
        node_count = degree(data.edge_label_index[0, mask] - start,  
                            num_nodes=logits.size(0))

        # 获取Top-K预测：相似度最高的K个图书索引
        topk_index = logits.topk(k, dim=-1).indices 
        # 检查Top-K预测是否在真实标签中
        isin_mat = ground_truth.gather(1, topk_index)  

        # 计算精确率：对于每个用户，Top-K中正确预测的数量除以K
        precision += float((isin_mat.sum(dim=-1) / k).sum()) 
        # 计算召回率：对于每个用户，Top-K中正确预测的数量除以该用户的测试边总数
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum()) 
        # 累加有效用户数（有测试边的用户）
        total_examples += int((node_count > 0).sum())  

    return precision / total_examples, recall / total_examples  


# ===================== 训练循环 =====================
for epoch in range(1, 101):  
    loss = train() # 调用训练函数，返回平均损失  
    precision, recall = test(k=20)  # 调用测试函数，计算Precision@20和Recall@20
    # 记录损失、精确率、召回率
    train_losses.append(loss)  
    precisions.append(precision)  
    recalls.append(recall)  
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@20: '
          f'{precision:.4f}, Recall@20: {recall:.4f}')  


# ===================== 绘图 =====================
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