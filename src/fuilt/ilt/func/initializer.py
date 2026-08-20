import math
import multiprocessing as mp

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

try:
    from .settings import *
    from . import utils as common
    from . import simple as lithosim
except ImportError:
    from settings import *
    import utils as common
    import simple as lithosim
# import pylitho.exact as lithosim

class Initializer: 
    def __init__(self): 
        pass
    
    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE): 
        pass

#直接将设计矩阵转换为张量
class PlainInit(Initializer): 
    def __init__(self): 
        super(PlainInit, self).__init__()
    
    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE): 
        if isinstance(design, glp.Design): 
            design = design.mat(sizeX, sizeY, offsetX, offsetY)#转化为numpy数组
        target = torch.tensor(design, dtype=dtype, device=device)
        params = target.clone()# 参数与目标完全相同
        return target, params

#对输入掩模进行归一化处理，将像素值从[0, 1]范围转换到[-1, 1]范围
class PixelInit(Initializer): 
    def __init__(self): 
        super(PixelInit, self).__init__()
    
    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE): 
        if isinstance(design, glp.Design): 
            design = design.mat(sizeX, sizeY, offsetX, offsetY)
        target = torch.tensor(design, dtype=dtype, device=device)
        params = target * 2.0 - 1.0 # 变化公式为 2 * x - 1，将[0, 1]范围的像素值转换为[-1, 1]范围
        return target, params


#距离场计算函数（numpy实现）
def _distMatPolygon(polygon, canvas, offsets): 
    if len(canvas) == 4: 
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    sizeX, sizeY = maxX - minX, maxY - minY
    #初始化距离场为极大值
    dist = np.ones([sizeX, sizeY]) * (sizeX * sizeY)
    xs = np.arange(minX, maxX, 1, dtype=np.int32).reshape([sizeX, 1])
    ys = np.arange(minY, maxY, 1, dtype=np.int32).reshape([1, sizeY])
    xs = np.tile(xs, [1, sizeY])
    ys = np.tile(ys, [sizeX, 1])
    
    frPt = polygon[-1]
    #遍历多边形的每条边
    for toPt in polygon: 
        frX, frY = frPt
        toX, toY = toPt
        if frX > toX: 
            frX, toX = toX, frX
        if frY > toY: 
            frY, toY = toY, frY
        frX += offsets[0]
        toX += offsets[0]
        frY += offsets[1]
        toY += offsets[1]
        #计算像素到边两个端点的距离
        dist1 = np.sqrt((frX - xs)**2 + (frY - ys)**2)
        dist2 = np.sqrt((toX - xs)**2 + (toY - ys)**2)
        
        dist = np.minimum(dist, np.minimum(dist1, dist2))
        #如果边是水平或垂直的，计算像素到边的距离
        if frX == toX: #垂直
            mask = (frY <= ys) * (ys <= toY)
            new = np.minimum(dist, np.abs(frX - xs))
            dist[mask] = new[mask]
        elif frY == toY: # 水平
            mask = (frX <= xs) * (xs <= toX)
            new = np.minimum(dist, np.abs(frY - ys))
            dist[mask] = new[mask]
            
        frPt = toPt
    return dist.T

#合并多个多边形的距离场（单线程遍历多边形）
def _distMatLegacy(design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512]): 
    if len(canvas) == 4: 
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    
    mask = design.mat(sizeX=maxX-minX, sizeY=maxY-minY, offsetX=offsets[0], offsetY=offsets[1])
    dist = np.ones([maxX-minX, maxY-minY]) * ((maxX-minX) * (maxY-minY))
    for polygon in design.polygons: 
        tmp = _distMatPolygon(polygon, canvas, offsets)
        dist = np.minimum(dist, tmp)
    dist[mask > 0] *= -1
    return dist


#距离场计算函数（torch实现）
def _distMatPolygonTorch(polygon, canvas, offsets): 
    if len(canvas) == 4: 
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    sizeX, sizeY = maxX - minX, maxY - minY

    dist = torch.ones([sizeX, sizeY], dtype=REALTYPE, device=DEVICE) * (sizeX * sizeY)
    xs = np.arange(minX, maxX, 1, dtype=np.int32).reshape([sizeX, 1])
    ys = np.arange(minY, maxY, 1, dtype=np.int32).reshape([1, sizeY])
    xs = torch.tensor(np.tile(xs, [1, sizeY]), dtype=REALTYPE, device=DEVICE)
    ys = torch.tensor(np.tile(ys, [sizeX, 1]), dtype=REALTYPE, device=DEVICE)
    
    frPt = polygon[-1]
    for toPt in polygon: 
        frX, frY = frPt
        toX, toY = toPt
        if frX > toX: 
            frX, toX = toX, frX
        if frY > toY: 
            frY, toY = toY, frY
        frX += offsets[0]
        toX += offsets[0]
        frY += offsets[1]
        toY += offsets[1]
        
        dist1 = torch.sqrt((frX - xs)**2 + (frY - ys)**2)
        dist2 = torch.sqrt((toX - xs)**2 + (toY - ys)**2)
        
        dist = torch.minimum(dist, torch.minimum(dist1, dist2))

        if frX == toX: 
            mask = (frY <= ys) * (ys <= toY)
            new = torch.minimum(dist, torch.abs(frX - xs))
            dist[mask] = new[mask]
        elif frY == toY: 
            mask = (frX <= xs) * (xs <= toX)
            new = torch.minimum(dist, torch.abs(frY - ys))
            dist[mask] = new[mask]
            
        frPt = toPt
    return dist.T

#合并多个多边形距离场（torch实现），并标记内部区域（负值）
def _distMatTorch(design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512], mask=None): 
    if len(canvas) == 4: 
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    
    if mask is None: 
        mask = design.mat(sizeX=maxX-minX, sizeY=maxY-minY, offsetX=offsets[0], offsetY=offsets[1])
    #初始化距离场为极大值
    dist = torch.ones([maxX-minX, maxY-minY], dtype=REALTYPE, device=DEVICE) * ((maxX-minX) * (maxY-minY))
    #叠加所有多边形的距离场
    for polygon in design.polygons: 
        tmp = _distMatPolygonTorch(polygon, canvas, offsets)
        dist = torch.minimum(dist, tmp)#保留最小距离
    #标记内部区域(掩模>0的区域取负值)
    dist[mask > 0] *= -1
    return dist#输出带符号的距离场（内部为负，外部为正）

#合并多个多边形的距离场（多线程遍历多边形）
def _distMat(design, canvas=[[0, 0], [2048, 2048]], offsets=[512, 512]): 
    if len(canvas) == 4: 
        canvas = [[canvas[0], canvas[1]], [canvas[2], canvas[3]]]
    minX, minY, maxX, maxY = canvas[0][0], canvas[0][1], canvas[1][0], canvas[1][1]
    
    pool = mp.Pool(processes=mp.cpu_count()//2)
    procs = []
    for polygon in design.polygons: 
        proc = pool.apply_async(_distMatPolygon, (polygon, canvas, offsets))
        procs.append(proc)
    pool.close()
    pool.join()

    dist = np.ones([maxX-minX, maxY-minY]) * ((maxX-minX) * (maxY-minY))
    for proc in procs: 
        tmp = proc.get()
        dist = np.minimum(dist, tmp)
    mask = design.mat(sizeX=maxX-minX, sizeY=maxY-minY, offsetX=offsets[0], offsetY=offsets[1])
    dist[mask > 0] *= -1
    
    return dist

class LevelSetInit(Initializer): 
    def __init__(self): 
        super(LevelSetInit, self).__init__()

    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE): 
        # 生成目标掩模（target）
        target = torch.tensor(design.mat(sizeX, sizeY, offsetX, offsetY), dtype=dtype, device=device)
        params = torch.tensor(
            _distMat(design, canvas=[[0, 0], [sizeX, sizeY]], offsets=[offsetX, offsetY]),
            dtype=REALTYPE, 
            device=DEVICE, 
            requires_grad=True # 启用梯度计算
        )
        return target, params

class LevelSetInitTorch(Initializer): 
    def __init__(self): 
        super(LevelSetInitTorch, self).__init__()

    def run(self, design, sizeX, sizeY, offsetX, offsetY, dtype=REALTYPE, device=DEVICE): 
        target = torch.tensor(design.mat(sizeX, sizeY, offsetX, offsetY), dtype=dtype, device=device)
        params = _distMatTorch(design, canvas=[[0, 0], [sizeX, sizeY]], offsets=[offsetX, offsetY]).detach().clone().requires_grad_(True)
        return target, params



class LevelSetImageInit(Initializer): 
    def __init__(self): 
        super(LevelSetImageInit, self).__init__()

    def run(self, mask, dtype=REALTYPE, device=DEVICE):
        """
        从二值图像生成LevelSet初始化 - 修复版本
        """
        # 确保mask是numpy数组
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # 归一化到0/1
        if mask.max() > 1.5:  # 如果是0/255格式
            mask = mask / 255.0
        
        mask = (mask > 0.5).astype(np.float32)
        sizeY, sizeX = mask.shape
        
        # print(f"  🔧 LevelSet初始化: mask尺寸={mask.shape}, 值域=[{mask.min():.1f}, {mask.max():.1f}]")
        
        # 🔧 使用更准确的距离变换
        # 内部距离（负值）
        inner_dist = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # 外部距离（正值）
        outer_dist = cv2.distanceTransform((mask < 0.5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        # 合并为带符号距离场
        signed_dist = outer_dist - inner_dist
        
        # 🔧 可选：轻微平滑距离场以减少数值问题
        signed_dist = cv2.GaussianBlur(signed_dist, (3, 3), 0.5)
        
        # print(f"  📊 距离场统计: 值域=[{signed_dist.min():.1f}, {signed_dist.max():.1f}]")
        # print(f"  📊 零水平集像素数: {np.sum(np.abs(signed_dist) < 1.0)}")
        
        # 转换为张量
        target_tensor = torch.tensor(mask, dtype=dtype, device=device)
        params = torch.tensor(signed_dist, dtype=dtype, device=device).requires_grad_(True)
        
        return target_tensor, params

# 🔧 备选方案：使用scipy的距离变换
class LevelSetImageInitScipy(Initializer): 
    def __init__(self): 
        super(LevelSetImageInitScipy, self).__init__()

    def run(self, mask, dtype=REALTYPE, device=DEVICE):
        """
        使用scipy的距离变换 - 更精确
        """
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            print("⚠️  scipy不可用，回退到OpenCV版本")
            return LevelSetImageInit().run(mask, dtype, device)
        
        # 确保mask是numpy数组
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # 归一化到0/1
        if mask.max() > 1.5:
            mask = mask / 255.0
        
        mask_bool = (mask > 0.5).astype(bool)
        
        print(f"  🔧 SciPy LevelSet初始化: mask尺寸={mask.shape}")
        
        # 使用scipy的精确距离变换
        inner_dist = distance_transform_edt(mask_bool)
        outer_dist = distance_transform_edt(~mask_bool)
        
        # 带符号距离场
        signed_dist = outer_dist - inner_dist
        
        # print(f"  📊 距离场统计: 值域=[{signed_dist.min():.1f}, {signed_dist.max():.1f}]")
        
        # 转换为张量
        target_tensor = torch.tensor(mask.astype(np.float32), dtype=dtype, device=device)
        params = torch.tensor(signed_dist.astype(np.float32), dtype=dtype, device=device).requires_grad_(True)
        
        return target_tensor, params

# 🔧 原始方法的修复版本（保持与原代码兼容）
class LevelSetImageInitFixed(Initializer):
    def __init__(self): 
        super(LevelSetImageInitFixed, self).__init__()

    def run(self, mask, dtype=REALTYPE, device=DEVICE):
        """
        修复原始距离场计算方法
        """
        # 确保mask是numpy数组
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        if mask.max() > 1.5:
            mask = mask / 255.0
        
        mask = (mask > 0.5).astype(np.float32)
        sizeY, sizeX = mask.shape
        
        print(f"  🔧 修复版LevelSet初始化: mask尺寸={mask.shape}")
        
        # 提取轮廓
        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            print("  ⚠️  没有找到轮廓，使用简单初始化")
            # 如果没有轮廓，使用简单的二值初始化
            target_tensor = torch.tensor(mask, dtype=dtype, device=device)
            params = torch.tensor((mask - 0.5) * 2.0, dtype=dtype, device=device).requires_grad_(True)
            return target_tensor, params
        
        # print(f"  📐 找到 {len(contours)} 个轮廓")
        
        # 初始化距离场
        dist = np.full((sizeY, sizeX), float('inf'), dtype=np.float32)
        
        # 创建坐标网格
        y_coords, x_coords = np.mgrid[0:sizeY, 0:sizeX]
        
        # 🔧 对每个轮廓计算真实的点到轮廓距离
        for contour in contours:
            if len(contour) < 3:
                continue
                
            # 简化轮廓点
            contour = contour.reshape(-1, 2)  # (N, 2)
            
            # 对每个像素计算到轮廓的最短距离
            for i in range(len(contour)):
                p1 = contour[i]
                p2 = contour[(i + 1) % len(contour)]
                
                # 计算到线段p1-p2的距离
                dist_to_segment = _point_to_segment_distance_vectorized(
                    x_coords, y_coords, p1[0], p1[1], p2[0], p2[1]
                )
                
                dist = np.minimum(dist, dist_to_segment)
        
        # 内部区域距离为负
        dist[mask > 0.5] *= -1
        
        # 处理无穷大值
        dist[dist == float('inf')] = max(sizeX, sizeY)
        dist[dist == float('-inf')] = -max(sizeX, sizeY)
        
        # print(f"  📊 距离场统计: 值域=[{dist.min():.1f}, {dist.max():.1f}]")
        
        # 转换为张量
        target_tensor = torch.tensor(mask, dtype=dtype, device=device)
        params = torch.tensor(dist, dtype=dtype, device=device).requires_grad_(True)
        
        return target_tensor, params

def _point_to_segment_distance_vectorized(px, py, x1, y1, x2, y2):
    """
    计算点到线段的距离（向量化版本）
    """
    # 线段向量
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        # 线段退化为点
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # 参数t表示投影点在线段上的位置
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = np.clip(t, 0, 1)  # 限制在线段范围内
    
    # 投影点坐标
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    # 距离
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

class LevelSetImageInitSmooth(Initializer): 
    def __init__(self, blur_kernel=(5, 5), blur_sigma=1.0, blur_times=2): 
        super(LevelSetImageInitSmooth, self).__init__()
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.blur_times = blur_times

    def run(self, mask, dtype=REALTYPE, device=DEVICE):
        """
        距离场多次高斯模糊，角点更平滑
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.max() > 1.5:
            mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)
        sizeY, sizeX = mask.shape

        # 内外距离场
        inner_dist = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        outer_dist = cv2.distanceTransform((mask < 0.5).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        signed_dist = outer_dist - inner_dist

        # 多次高斯模糊
        for _ in range(self.blur_times):
            signed_dist = cv2.GaussianBlur(signed_dist, self.blur_kernel, self.blur_sigma)

        # 转为tensor
        target_tensor = torch.tensor(mask, dtype=dtype, device=device)
        params = torch.tensor(signed_dist, dtype=dtype, device=device).requires_grad_(True)
        return target_tensor, params

# 推荐用法：在levelset.py里替换为
# target, params = initializer.LevelSetImageInitSmooth().run(mask)