import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

# 加载数据
dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0]
num_classes = dataset.num_classes
num_features = dataset.num_features

class GlobalContextSemantic(nn.Module):
    def __init__(self,
                 in_channels=num_features,  # 输入特征维度（Cora数据集默认）
                 hidden_channels=16,
                 num_classes=num_classes,  # Cora类别数
                 gcn_layers=2):  # GCN层数
        super().__init__()
        self.convs = torch.nn.ModuleList()

        # 动态构建GCN层
        for i in range(gcn_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNConv(in_dim, hidden_channels))

        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        """
        Args:
            x: Input node features of shape (N, C), N为节点数，C为特征维度。
        Returns:
            Updated node features of shape (N, C).
        """
        N, C = data.shape

        # 计算查询和键的变换
        query = self.W_theta(data)  # (N, C)
        key = self.W_phi(data)  # (N, C)

        # 计算相似性矩阵（嵌入高斯函数）
        sim_matrix = torch.matmul(query, key.transpose(0, 1))  # (N, N)
        adj_matrix = F.softmax(sim_matrix, dim=-1)  # Softmax归一化

        # 多头图注意力传播
        x_multihead = []
        for k in range(self.num_heads):
            # 线性变换 + 邻接矩阵传播
            x_k = self.W_k[k](data)  # (N, C)
            x_k = torch.matmul(adj_matrix, x_k)  # (N, C)
            x_k = F.relu(x_k)
            x_multihead.append(x_k)

        # 拼接多头输出并平均
        x_updated = torch.stack(x_multihead, dim=0).mean(dim=0)  # (N, C)

        return x_updated

# 训练和测试
model = GlobalContextSemantic()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    return acc


if __name__ == '__main__':

    for epoch in range(200):
        train()
        test_acc = test()
        if epoch % 10 == 0:  # 每10个epoch打印一次
            print(f'Epoch: {epoch}, Test Acc: {test_acc:.4f}')