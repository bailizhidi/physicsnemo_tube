import os
import glob
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch_geometric as pyg
from physicsnemo.datapipes.gnn.utils import load_json, save_json

class DeformingPlateDataset(Dataset):
    def __init__(self, name="dataset", data_dir=None, split="train", num_samples=4, num_steps=91, noise_std=0.003):
        self.name = name
        self.data_dir = data_dir
        self.split = split
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.noise_std = noise_std
        self.length = num_samples * (num_steps - 1)

        print(f"Preparing the {split} dataset from .npz files...")

        # --- 核心修改 1: 直接读取 .npz 文件夹 ---
        self.sample_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if len(self.sample_files) < num_samples:
            print(f"⚠️ Warning: Only found {len(self.sample_files)} files.")
            self.num_samples = len(self.sample_files)
            self.length = self.num_samples * (num_steps - 1)

        self.graphs, self.cells, self.node_type, self.thickness = [], [], [], []
        self.moving_points_mask, self.object_points_mask, self.clamped_points_mask = [], [], []
        self.mesh_pos = []
        noise_mask = []

        for i, file_path in enumerate(self.sample_files[:self.num_samples]):
            data_np = np.load(file_path)

            # 解析四边形网格拓扑
            src, dst = self.cell_to_adj(data_np["cells"])
            graph = self.create_graph(src, dst, dtype=torch.int32)
            graph = self.add_edge_features(graph, data_np["mesh_pos"])
            self.graphs.append(graph)

            node_type = torch.tensor(data_np["node_type"], dtype=torch.uint8)
            self.node_type.append(self._one_hot_encode(node_type))
            noise_mask.append(torch.eq(node_type, torch.zeros_like(node_type)))

            # 提取厚度特征 [N, 1]
            self.thickness.append(torch.tensor(data_np["thickness"], dtype=torch.float32))

            if self.split != "train":
                self.mesh_pos.append(torch.tensor(data_np["mesh_pos"]))
                self.cells.append(data_np["cells"])
                m_mask, o_mask, c_mask = self._get_rollout_mask(node_type)
                self.moving_points_mask.append(m_mask)
                self.object_points_mask.append(o_mask)
                self.clamped_points_mask.append(c_mask)

        # 边特征归一化
        if self.split == "train":
            self.edge_stats = self._get_edge_stats()
        else:
            self.edge_stats = load_json("edge_stats.json")

        for i in range(self.num_samples):
            self.graphs[i].edge_attr = self.normalize_edge(
                self.graphs[i], self.edge_stats["edge_mean"], self.edge_stats["edge_std"]
            )

        self.node_features, self.node_targets = [], []
        for i, file_path in enumerate(self.sample_files[:self.num_samples]):
            data_np = np.load(file_path)
            features, targets = {}, {}

            world_pos = data_np["world_pos"][:num_steps]
            stress = data_np["stress"][:num_steps]
            strain = data_np["strain"][:num_steps]
            peeq = data_np["peeq"][:num_steps]

            features["world_pos"] = self._drop_last(world_pos)
            targets["velocity"] = self._push_forward_diff(world_pos)
            targets["stress"] = self._push_forward(stress)
            targets["strain"] = self._push_forward(strain)
            targets["peeq"] = self._push_forward(peeq)

            if split == "train":
                features["world_pos"], targets["velocity"] = self._add_noise(
                    features["world_pos"], targets["velocity"], self.noise_std, noise_mask[i]
                )

            self.node_features.append(features)
            self.node_targets.append(targets)

        # --- 核心修改 2: 更新 node stats 包含新物理场 ---
        if self.split == "train":
            self.node_stats = self._get_node_stats()
        else:
            self.node_stats = load_json("node_stats.json")

        for i in range(self.num_samples):
            self.node_targets[i]["velocity"] = self.normalize_node(self.node_targets[i]["velocity"], self.node_stats["velocity_mean"], self.node_stats["velocity_std"])
            self.node_targets[i]["stress"] = self.normalize_node(self.node_targets[i]["stress"], self.node_stats["stress_mean"], self.node_stats["stress_std"])
            self.node_targets[i]["strain"] = self.normalize_node(self.node_targets[i]["strain"], self.node_stats["strain_mean"], self.node_stats["strain_std"])
            self.node_targets[i]["peeq"] = self.normalize_node(self.node_targets[i]["peeq"], self.node_stats["peeq_mean"], self.node_stats["peeq_std"])

    def __getitem__(self, idx):
        gidx = idx // (self.num_steps - 1)
        tidx = idx % (self.num_steps - 1)
        graph = self.graphs[gidx].clone()

        # --- 核心修改 3: 输入特征拼接 厚度(Thickness) ---
        node_features = torch.cat((self.node_type[gidx].float(), self.thickness[gidx]), dim=-1)

        # --- 核心修改 4: 目标特征拼接应变和PEEQ ---
        node_targets = torch.cat((
            self.node_targets[gidx]["velocity"][tidx],
            self.node_targets[gidx]["stress"][tidx],
            self.node_targets[gidx]["strain"][tidx],
            self.node_targets[gidx]["peeq"][tidx]
        ), dim=-1)

        graph.x = node_features
        graph.y = node_targets
        graph.world_pos = self.node_features[gidx]["world_pos"][tidx]

        if self.split == "train":
            return graph
        else:
            graph.mesh_pos = self.mesh_pos[gidx]
            cells = torch.as_tensor(self.cells[gidx])
            moving_points_mask = self.moving_points_mask[gidx]
            object_points_mask = self.object_points_mask[gidx]
            clamped_points_mask = self.clamped_points_mask[gidx]
            return graph, cells, moving_points_mask, object_points_mask, clamped_points_mask

    def __len__(self):
        return self.length

    def _get_edge_stats(self):
        stats = {"edge_mean": 0, "edge_meansqr": 0}
        for i in range(self.num_samples):
            stats["edge_mean"] += torch.mean(self.graphs[i].edge_attr, dim=0) / self.num_samples
            stats["edge_meansqr"] += torch.mean(torch.square(self.graphs[i].edge_attr), dim=0) / self.num_samples
        stats["edge_std"] = torch.sqrt(stats["edge_meansqr"] - torch.square(stats["edge_mean"]))
        stats.pop("edge_meansqr")
        save_json(stats, "edge_stats.json")
        return stats

    def _get_node_stats(self):
        stats = {
            "velocity_mean": 0, "velocity_meansqr": 0,
            "stress_mean": 0, "stress_meansqr": 0,
            "strain_mean": 0, "strain_meansqr": 0,
            "peeq_mean": 0, "peeq_meansqr": 0,
        }
        for i in range(self.num_samples):
            stats["velocity_mean"] += torch.mean(self.node_targets[i]["velocity"], dim=(0, 1)) / self.num_samples
            stats["velocity_meansqr"] += torch.mean(torch.square(self.node_targets[i]["velocity"]), dim=(0, 1)) / self.num_samples
            stats["stress_mean"] += torch.mean(self.node_targets[i]["stress"], dim=(0, 1)) / self.num_samples
            stats["stress_meansqr"] += torch.mean(torch.square(self.node_targets[i]["stress"]), dim=(0, 1)) / self.num_samples
            stats["strain_mean"] += torch.mean(self.node_targets[i]["strain"], dim=(0, 1)) / self.num_samples
            stats["strain_meansqr"] += torch.mean(torch.square(self.node_targets[i]["strain"]), dim=(0, 1)) / self.num_samples
            stats["peeq_mean"] += torch.mean(self.node_targets[i]["peeq"], dim=(0, 1)) / self.num_samples
            stats["peeq_meansqr"] += torch.mean(torch.square(self.node_targets[i]["peeq"]), dim=(0, 1)) / self.num_samples

        stats["velocity_std"] = torch.sqrt(stats["velocity_meansqr"] - torch.square(stats["velocity_mean"]))
        stats["stress_std"] = torch.sqrt(stats["stress_meansqr"] - torch.square(stats["stress_mean"]))
        stats["strain_std"] = torch.sqrt(stats["strain_meansqr"] - torch.square(stats["strain_mean"]))
        stats["peeq_std"] = torch.sqrt(stats["peeq_meansqr"] - torch.square(stats["peeq_mean"]))
        
        # 防止全是0的场导致除以0错误
        stats["stress_std"] = torch.where(stats["stress_std"] < 1e-6, torch.ones_like(stats["stress_std"]), stats["stress_std"])
        stats["strain_std"] = torch.where(stats["strain_std"] < 1e-6, torch.ones_like(stats["strain_std"]), stats["strain_std"])
        stats["peeq_std"] = torch.where(stats["peeq_std"] < 1e-6, torch.ones_like(stats["peeq_std"]), stats["peeq_std"])

        for k in ["velocity_meansqr", "stress_meansqr", "strain_meansqr", "peeq_meansqr"]:
            stats.pop(k)

        save_json(stats, "node_stats.json")
        return stats

    @staticmethod
    def cell_to_adj(cells):
        """核心修改 5: 适配 Abaqus S4R 四边形壳单元"""
        num_cells = np.shape(cells)[0]
        # 四边形的四条物理边界
        edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0)]
        src = [cells[i][a] for i in range(num_cells) for a, b in edge_indices]
        dst = [cells[i][b] for i in range(num_cells) for a, b in edge_indices]
        return src, dst

    @staticmethod
    def create_graph(src, dst, dtype=torch.int32):
        edges = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0).long()
        edges = pyg.utils.to_undirected(edges)
        edges = pyg.utils.coalesce(edges)
        if isinstance(edges, tuple): edges = edges[0]
        return pyg.data.Data(edge_index=edges)

    @staticmethod
    def add_edge_features(graph, pos):
        row, col = graph.edge_index
        disp = torch.tensor(pos[row] - pos[col])
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        graph.edge_attr = torch.cat((disp, disp_norm), dim=1)
        return graph

    @staticmethod
    def normalize_node(invar, mu, std):
        return (invar - mu.expand(invar.size())) / std.expand(invar.size())

    @staticmethod
    def normalize_edge(graph, mu, std):
        return (graph.edge_attr - mu) / std

    @staticmethod
    def denormalize(invar, mu, std):
        return invar * std + mu

    @staticmethod
    def _one_hot_encode(node_type):
        node_type = torch.squeeze(node_type, dim=-1)
        mapping = {0: 0, 1: 1, 3: 2}
        mapped = torch.full_like(node_type, fill_value=-1)
        for k, v in mapping.items():
            mapped[node_type == k] = v
        node_type = F.one_hot(mapped.long(), num_classes=3)
        return node_type

    @staticmethod
    def _drop_last(invar): return torch.tensor(invar[0:-1], dtype=torch.float)

    @staticmethod
    def _push_forward(invar): return torch.tensor(invar[1:], dtype=torch.float)

    @staticmethod
    def _push_forward_diff(invar): return torch.tensor(invar[1:] - invar[0:-1], dtype=torch.float)

    @staticmethod
    def _get_rollout_mask(node_type):
        return (torch.eq(node_type, torch.zeros_like(node_type)),
                torch.eq(node_type, torch.zeros_like(node_type) + 1),
                torch.eq(node_type, torch.zeros_like(node_type) + 3))

    @staticmethod
    def _add_noise(features, targets, noise_std, noise_mask):
        noise = torch.normal(mean=0, std=noise_std, size=features.size())
        noise_mask = noise_mask.expand(features.size()[0], -1, 3)
        noise = torch.where(noise_mask, noise, torch.zeros_like(noise))
        features += noise
        targets -= noise
        return features, targets