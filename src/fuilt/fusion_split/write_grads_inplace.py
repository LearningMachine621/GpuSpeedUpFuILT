import math
from typing import List, Tuple
import torch

class StaticFusionSplitManager:
    def __init__(self, base_grid: int, s0: int, overlap: int, device: torch.device):
        """
        初始化静态工作区。这里面的所有 torch.zeros 只会在脚本刚开始跑一次。
        """
        self.base_grid = base_grid
        self.s0 = s0
        self.p0 = s0 - overlap
        self.canvas_size = s0 + (base_grid - 1) * self.p0
        
        print(f"🧩 初始化静态工作区: 画布大小 {self.canvas_size}x{self.canvas_size}, 正在预分配显存...")

        # 1. 唯一的大图 Buffer (Fuse 的主体)
        self.global_fused_buffer = torch.zeros(
            (self.canvas_size, self.canvas_size), 
            dtype=torch.float32, 
            device=device
        )
        
        # 2. 预计算每个像素的重叠次数 (Count)
        count_buffer = torch.zeros_like(self.global_fused_buffer)
        self.coords: List[Tuple[int, int]] = []
        
        for r in range(base_grid):
            for c in range(base_grid):
                y0 = r * self.p0
                x0 = c * self.p0
                self.coords.append((y0, x0))
                # 累加覆盖次数
                count_buffer[y0:y0+s0, x0:x0+s0] += 1.0
                
        # 将除法转化为乘法掩码 (1.0 / count)，处理除 0 异常保底
        count_buffer[count_buffer == 0] = 1.0 
        self.inv_count_buffer = 1.0 / count_buffer

        # 3. 预分配 Split 接收的小图 Buffer 列表
        self.split_grads_buffers = [
            torch.zeros((s0, s0), dtype=torch.float32, device=device) 
            for _ in range(base_grid * base_grid)
        ]

    def fuse(self, grads: List[torch.Tensor]) -> torch.Tensor:
        """极致 In-place Fuse，显存分配量为 0"""
        if len(grads) != len(self.coords):
            raise ValueError(f"grads 数量不匹配: {len(grads)} vs {len(self.coords)}")

        # 把大黑板擦干净 (原地清零)
        self.global_fused_buffer.zero_()  
        
        # 原地累加所有小 Tile
        for i, (y0, x0) in enumerate(self.coords):
            self.global_fused_buffer[y0:y0+self.s0, x0:x0+self.s0].add_(grads[i])
            
        # 原地乘法求平均 (替代除法)
        self.global_fused_buffer.mul_(self.inv_count_buffer)
        
        return self.global_fused_buffer

    def split(self, top: torch.Tensor) -> List[torch.Tensor]:
        """极致 In-place Split，显存分配量为 0"""
        if top.shape != self.global_fused_buffer.shape:
            raise ValueError(
                f"top 尺寸不匹配: {tuple(top.shape)} vs {tuple(self.global_fused_buffer.shape)}"
            )

        for i, (y0, x0) in enumerate(self.coords):
            # 将大图对应区域的数值，原地拷贝进提前挖好的小坑里
            self.split_grads_buffers[i].copy_(top[y0:y0+self.s0, x0:x0+self.s0])
            
        return self.split_grads_buffers

    def fuse_batched(self, batched_grads: torch.Tensor) -> torch.Tensor:
        """接收 [B, H, W]，原地融合到大图。"""
        if batched_grads.dim() != 3:
            raise ValueError(f"batched_grads 维度错误: {batched_grads.dim()}，期望 3")
        if batched_grads.shape[0] != len(self.coords):
            raise ValueError(f"batch 数量不匹配: {batched_grads.shape[0]} vs {len(self.coords)}")
        if batched_grads.shape[1] != self.s0 or batched_grads.shape[2] != self.s0:
            raise ValueError(
                f"单图尺寸不匹配: {tuple(batched_grads.shape[1:])} vs ({self.s0}, {self.s0})"
            )

        self.global_fused_buffer.zero_()
        for i, (y0, x0) in enumerate(self.coords):
            self.global_fused_buffer[y0:y0+self.s0, x0:x0+self.s0].add_(batched_grads[i])
        self.global_fused_buffer.mul_(self.inv_count_buffer)
        return self.global_fused_buffer

    def split_batched(self, top: torch.Tensor, out_batch: torch.Tensor) -> None:
        """把大图切片，原地填入 [B, H, W]。"""
        if top.shape != self.global_fused_buffer.shape:
            raise ValueError(f"top 尺寸不匹配: {tuple(top.shape)} vs {tuple(self.global_fused_buffer.shape)}")
        if out_batch.dim() != 3:
            raise ValueError(f"out_batch 维度错误: {out_batch.dim()}，期望 3")
        if out_batch.shape[0] != len(self.coords):
            raise ValueError(f"out_batch batch 数量不匹配: {out_batch.shape[0]} vs {len(self.coords)}")
        if out_batch.shape[1] != self.s0 or out_batch.shape[2] != self.s0:
            raise ValueError(
                f"out_batch 单图尺寸不匹配: {tuple(out_batch.shape[1:])} vs ({self.s0}, {self.s0})"
            )

        for i, (y0, x0) in enumerate(self.coords):
            out_batch[i].copy_(top[y0:y0+self.s0, x0:x0+self.s0])