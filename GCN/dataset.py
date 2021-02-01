import torch
import torchvision

# ノードのデータセットと捉える


class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, nodedataset, edgedataset, node_adj, edge_adj, D_v, D_e, T):
        self.transform1 = torchvision.transforms.ToTensor()
        self.nodedataset = nodedataset  # data_num*node_num*channel
        self.edgedataset = edgedataset  # data_num*edge_num*channel
        self.node_adj = node_adj
        self.edge_adj = edge_adj
        self.D_v = D_v
        self.D_e = D_e
        self.T = T

        self.datanum = len(nodedataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        node_data = torch.from_numpy(self.nodedataset[idx]).clone().double()
        edge_data = torch.from_numpy(self.edgedataset[idx]).clone().double()
        node_adj = torch.from_numpy(self.node_adj).clone().double()
        edge_adj = torch.from_numpy(self.edge_adj).clone().double()
        D_v = torch.from_numpy(self.D_v).clone().double()
        D_e = torch.from_numpy(self.D_e).clone().double()
        T = torch.from_numpy(self.T).clone().double()

        return node_data, edge_data, node_adj, edge_adj, D_v, D_e, T
