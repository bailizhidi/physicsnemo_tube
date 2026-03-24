# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import torch
import vtk
from vtk.util import numpy_support
from torch.utils.data import DataLoader
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from physicsnemo.models.meshgraphnet import HybridMeshGraphNet
from deforming_plate_dataset import DeformingPlateDataset
from physicsnemo.utils.logging import PythonLogger
from physicsnemo.utils import load_checkpoint
from helpers import add_world_edges

def save_vtu(filename, points, cells, point_data_dict):
    """使用 VTK 库将四边形网格和物理场导出为 .vtu 文件"""
    ugrid = vtk.vtkUnstructuredGrid()

    # 1. 写入几何节点 (使用预测的 3D 坐标)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points, deep=True, array_type=vtk.VTK_FLOAT))
    ugrid.SetPoints(vtk_points)

    # 2. 写入四边形网格单元 (VTK_QUAD = 9)
    num_cells = cells.shape[0]
    cell_array = vtk.vtkCellArray()
    for i in range(num_cells):
        cell_array.InsertNextCell(4)
        for j in range(4):
            cell_array.InsertCellPoint(cells[i, j])
    ugrid.SetCells(vtk.VTK_QUAD, cell_array)

    # 3. 写入所有的标量/矢量物理场 (应力、应变、误差等)
    for name, data in point_data_dict.items():
        # 如果是 1D 数组，转为 [N, 1] 避免 VTK 报错
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        vtk_array = numpy_support.numpy_to_vtk(data, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_array.SetName(name)
        ugrid.GetPointData().AddArray(vtk_array)

    # 4. 保存文件
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(ugrid)
    writer.Write()

class MGNRollout:
    def __init__(self, cfg: DictConfig, logger: PythonLogger):
        self.num_test_time_steps = cfg.num_test_time_steps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")

        self.dataset = DeformingPlateDataset(
            name="deforming_plate_test",
            data_dir=to_absolute_path(cfg.data_dir),
            split="test",
            num_samples=cfg.num_test_samples,
            num_steps=cfg.num_test_time_steps,
        )

        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0])

        self.model = HybridMeshGraphNet(
            cfg.num_input_features, cfg.num_edge_features, cfg.num_output_features,
            mlp_activation_fn="silu" if cfg.recompute_activation else "relu",
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        load_checkpoint(to_absolute_path(cfg.ckpt_path), models=self.model, device=self.device)
        self.logger = logger

    @torch.inference_mode()
    def predict_and_export_vtu(self):
        output_dir = "vtk_output"
        os.makedirs(output_dir, exist_ok=True)
        stats = {k: v.to(self.device) for k, v in self.dataset.node_stats.items()}
        
        sample_idx = 0
        frame_idx = 0
        current_sample_dir = ""

        # 初始化历史位置记录器
        last_pred_pos = None

        for i, (graph, cells, moving_mask, obj_mask, clamp_mask) in enumerate(self.dataloader):
            graph = graph.to(self.device)
            moving_mask = moving_mask.to(self.device)
            obj_mask = obj_mask.to(self.device)
            clamp_mask = clamp_mask.to(self.device)

            # 新样本重置判断
            if i % (self.num_test_time_steps - 1) == 0:
                frame_idx = 1 # 第0帧是初始直线，直接从第1帧开始存
                current_sample_dir = os.path.join(output_dir, f"sample_{sample_idx}")
                os.makedirs(current_sample_dir, exist_ok=True)
                sample_idx += 1
                last_pred_pos = None
                self.logger.info(f"Processing Test Sample {sample_idx}...")

            # 提取真值目标
            exact_vel = self.dataset.denormalize(graph.y[:, 0:3], stats["velocity_mean"], stats["velocity_std"])
            exact_stress = self.dataset.denormalize(graph.y[:, 3:4], stats["stress_mean"], stats["stress_std"])
            exact_strain = self.dataset.denormalize(graph.y[:, 4:5], stats["strain_mean"], stats["strain_std"])
            exact_peeq = self.dataset.denormalize(graph.y[:, 5:6], stats["peeq_mean"], stats["peeq_std"])
            
            exact_next_pos = exact_vel + graph.world_pos[:, 0:3]

            # 注入上一帧的预测坐标
            if last_pred_pos is not None:
                graph.world_pos = last_pred_pos

            # 加载边并预测
            graph, mesh_edge, world_edge = add_world_edges(graph)
            pred = self.model(graph.x, mesh_edge, world_edge, graph)

            # 预测值反归一化
            pred_vel = self.dataset.denormalize(pred[:, 0:3], stats["velocity_mean"], stats["velocity_std"])
            pred_stress = self.dataset.denormalize(pred[:, 3:4], stats["stress_mean"], stats["stress_std"])
            pred_strain = self.dataset.denormalize(pred[:, 4:5], stats["strain_mean"], stats["strain_std"])
            pred_peeq = self.dataset.denormalize(pred[:, 5:6], stats["peeq_mean"], stats["peeq_std"])

            # 屏蔽固定边界的预测速度
            mask_3d = torch.cat((moving_mask, moving_mask, moving_mask), dim=-1).to(self.device)
            pred_vel = torch.where(mask_3d, pred_vel, torch.zeros_like(pred_vel))

            # 坐标积分与边界强覆盖 (Masking)
            pred_next_pos = pred_vel.squeeze(0) + graph.world_pos[:, 0:3]
            pred_next_pos = torch.where(obj_mask, exact_next_pos, pred_next_pos)
            pred_next_pos = torch.where(clamp_mask, exact_next_pos, pred_next_pos)

            # 记录历史供下一次使用
            last_pred_pos = pred_next_pos.squeeze(0).clone()

            # ==================================
            # VTK 数据打包与计算误差
            # ==================================
            points = pred_next_pos.cpu().numpy()
            c_cells = cells.cpu().numpy()

            p_pos = points
            e_pos = exact_next_pos.cpu().numpy()
            
            p_stress = pred_stress.cpu().numpy()
            e_stress = exact_stress.cpu().numpy()
            
            p_strain = pred_strain.cpu().numpy()
            e_strain = exact_strain.cpu().numpy()
            
            p_peeq = pred_peeq.cpu().numpy()
            e_peeq = exact_peeq.cpu().numpy()

            # 提取节点类型用于可视化区分管身和模具
            node_type = torch.argmax(graph.x[:, 0:3], dim=1).cpu().numpy()

            point_data = {
                "Pred_Stress (MPa)": p_stress,
                "Exact_Stress (MPa)": e_stress,
                "Error_Stress": np.abs(p_stress - e_stress),
                
                "Pred_Strain": p_strain,
                "Exact_Strain": e_strain,
                "Error_Strain": np.abs(p_strain - e_strain),
                
                "Pred_PEEQ": p_peeq,
                "Exact_PEEQ": e_peeq,
                "Error_PEEQ": np.abs(p_peeq - e_peeq),
                
                "Exact_Position_Vector": e_pos - p_pos, # 指向真实坐标的向量
                "Error_Displacement (mm)": np.linalg.norm(p_pos - e_pos, axis=1),
                
                "Node_Type": node_type
            }

            vtu_filename = os.path.join(current_sample_dir, f"frame_{frame_idx:03d}.vtu")
            save_vtu(vtu_filename, points, c_cells, point_data)
            
            frame_idx += 1

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")
    logger.file_logging()
    logger.info("Starting High-Performance VTU Export...")
    rollout = MGNRollout(cfg, logger)
    rollout.predict_and_export_vtu()
    logger.info("✅ All VTU files exported to the 'vtk_output' directory! Ready for ParaView.")

if __name__ == "__main__":
    main()