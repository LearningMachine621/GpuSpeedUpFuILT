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
        self._is_canvas_finalized = False

    def reset_canvas(self) -> None:
        """每轮迭代前，把大画布擦干净。"""
        self.global_fused_buffer.zero_()
        self._is_canvas_finalized = False

    def add_tile_grad_inplace(self, tile_idx: int, grad: torch.Tensor) -> None:
        """算出一块梯度后，直接原地累加到大画布。"""
        if self._is_canvas_finalized:
            raise RuntimeError("当前画布已 finalize，请先 reset_canvas 再继续 add")
        if tile_idx < 0 or tile_idx >= len(self.coords):
            raise IndexError(f"tile_idx 越界: {tile_idx}")
        if grad.shape != (self.s0, self.s0):
            raise ValueError(f"grad 尺寸不匹配: {tuple(grad.shape)} vs ({self.s0}, {self.s0})")
        if grad.dtype != torch.float32:
            raise ValueError(f"grad dtype 必须是 float32，当前: {grad.dtype}")
        if grad.device != self.global_fused_buffer.device:
            raise ValueError(
                f"grad device 不匹配: {grad.device} vs {self.global_fused_buffer.device}"
            )
        y0, x0 = self.coords[tile_idx]
        self.global_fused_buffer[y0:y0+self.s0, x0:x0+self.s0].add_(grad)

    def finalize_fuse(self) -> torch.Tensor:
        """所有小块贴完后，统一按覆盖次数做平均。"""
        if self._is_canvas_finalized:
            return self.global_fused_buffer
        self.global_fused_buffer.mul_(self.inv_count_buffer)
        self._is_canvas_finalized = True
        return self.global_fused_buffer

    def fuse(self, grads: List[torch.Tensor]) -> torch.Tensor:
        """极致 In-place Fuse，显存分配量为 0"""
        if len(grads) != len(self.coords):
            raise ValueError(f"grads 数量不匹配: {len(grads)} vs {len(self.coords)}")

        self.reset_canvas()
        for i, grad in enumerate(grads):
            self.add_tile_grad_inplace(i, grad)
        return self.finalize_fuse()

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

        self.reset_canvas()
        for i in range(batched_grads.shape[0]):
            self.add_tile_grad_inplace(i, batched_grads[i])
        return self.finalize_fuse()

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