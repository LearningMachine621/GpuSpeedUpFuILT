import os
import sys
import time
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

FILE_DIR = Path(__file__).resolve().parent
ILT_DIR = FILE_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LEVELSET_CONFIG = PROJECT_ROOT / "config" / "pylevelset1024.txt"
DEFAULT_LITHO_CONFIG = PROJECT_ROOT / "config" / "lithosimple.txt"


def _resolve_path(path_value):
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj

    raw = str(path_value).replace("\\", "/")
    if raw.startswith("./"):
        raw = raw[2:]

    project_candidate = PROJECT_ROOT / raw
    if project_candidate.exists() or raw.startswith(("config/", "output/", "visualizations", "tmp/")):
        return project_candidate
    return (ILT_DIR / raw).resolve()


try:
    from . import settings as _settings
    from . import utils as common
    from . import simple as lithosim
    from . import initializer
    from ...pre_work.read_oas2mask import read_oas_to_real_size_mask
except ImportError:
    import settings as _settings
    import utils as common
    import simple as lithosim
    import initializer
    from fuilt.pre_work.read_oas2mask import read_oas_to_real_size_mask

REALTYPE = _settings.REALTYPE
COMPLEXTYPE = _settings.COMPLEXTYPE
DEVICE = _settings.DEVICE






class LevelSetCfg: 
    def __init__(self, config): 
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        elif isinstance(config, str): 
            self._config = common.parseConfig(str(_resolve_path(config)))
        required = ["Iterations", "TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBL2", "WeightPVBand", "StepSize", 
                    "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in required: 
            assert key in self._config, f"[SimpleILT]: Cannot find the config {key}."
        intfields = ["Iterations", "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBL2", "WeightPVBand", "StepSize"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])
    
    def __getitem__(self, key): 
        return self._config[key]

def gradImage(image): 
    GRAD_STEPSIZE = 1.0
    image = image.view([-1, 1, image.shape[-2], image.shape[-1]])
    padded = func.pad(image, (1, 1, 1, 1), mode='replicate')[:, 0].detach()
    gradX = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * GRAD_STEPSIZE)
    gradY = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * GRAD_STEPSIZE)
    return gradX.view(image.shape), gradY.view(image.shape)
    
class _Binarize(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, levelset): 
        ctx.save_for_backward(levelset)
        mask = torch.zeros_like(levelset)
        mask[levelset < 0] = 1.0
        return mask
    
    @staticmethod
    def backward(ctx, *grad_outputs): 
        grad_output = grad_outputs[0]
        levelset, = ctx.saved_tensors
        gradX, gradY = gradImage(levelset)
        l2norm = torch.sqrt(gradX**2 + gradY**2)
        return -l2norm * grad_output
    
class Binarize(nn.Module): 
    def __init__(self): 
        super(Binarize, self).__init__()
        pass

    def forward(self, levelset): 
        return _Binarize.apply(levelset)

class LevelSet(nn.Module): 
    def __init__(self, lithosim): 
        super(LevelSet, self).__init__()
        self._binarize = Binarize()
        self._lithosim = lithosim

    def forward(self, params): 
        mask = self._binarize(params)
        printedNom, printedMax, printedMin = self._lithosim(mask)
        return mask, printedNom, printedMax, printedMin

class LevelSetILT: 
    def __init__(self, config=None, lithosim_model=None, device=DEVICE, multigpu=False): 
        super(LevelSetILT, self).__init__()
        
        # 🔧 修复默认参数问题
        if config is None:
            config = LevelSetCfg(str(DEFAULT_LEVELSET_CONFIG))
        if lithosim_model is None:
            lithosim_model = lithosim.LithoSim(str(DEFAULT_LITHO_CONFIG))
            
        self._config = config
        self._device = device
        
        # LevelSet
        self._levelset = LevelSet(lithosim_model).to(device)
        if multigpu: 
            self._levelset = nn.DataParallel(self._levelset)
            
        # Filter
        self._filter = torch.zeros([self._config["TileSizeX"], self._config["TileSizeY"]], dtype=REALTYPE, device=self._device)
        self._filter[self._config["OffsetX"]:self._config["OffsetX"]+self._config["ILTSizeX"], \
                     self._config["OffsetY"]:self._config["OffsetY"]+self._config["ILTSizeY"]] = 1
        
        # 新增：上一步梯度和方向
        self._prev_grad = None
        self._prev_dir = None

        # 新增：字典形式的历史梯度和方向
        self._prev_grad_dict = {}
        self._prev_dir_dict = {}
    
    def solve(self, target, params, curv=None, verbose=0): 
        # Initialize
        backup = params
        params = params.clone().detach().requires_grad_(True)

        # Optimizer 
        opt = optim.Adam([params], lr=self._config["StepSize"])
        
        # Optimization process
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None
        for idx in range(self._config["Iterations"]): 
            mask, printedNom, printedMax, printedMin = self._levelset(params * self._filter + backup * (1.0 - self._filter))
            l2loss = func.mse_loss(printedNom, target, reduction="sum")
            pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
            pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            pvband = torch.sum((printedMax >= self._config["TargetDensity"]) != (printedMin >= self._config["TargetDensity"]))
            loss = l2loss + self._config["WeightPVBL2"] * pvbl2 + self._config["WeightPVBand"] * pvbloss
            if not curv is None: 
                kernelCurv = torch.tensor([[-1.0/16, 5.0/16, -1.0/16], [5.0/16, -1.0, 5.0/16], [-1.0/16, 5.0/16, -1.0/16]], dtype=REALTYPE, device=DEVICE)
                curvature = func.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
                losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
                loss += curv * losscurv
            if verbose == 1: 
                print(f"[Iteration {idx}]: L2 = {l2loss.item():.0f}; PVBand: {pvband.item():.0f}")

            if bestParams is None or bestMask is None or loss.item() < lossMin: 
                lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
                bestParams = params.detach().clone()
                bestMask = mask.detach().clone()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return l2Min, pvbMin, bestParams, bestMask
    
    def solve_sacg(self, target, params, curv=None, verbose=0):
        # 初始化
        backup = params
        params = params.clone().detach().requires_grad_(True)

        # 记录初始梯度
        mask, printedNom, printedMax, printedMin = self._levelset(params * self._filter + backup * (1.0 - self._filter))
        l2loss = func.mse_loss(printedNom, target, reduction="sum")
        pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
        pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
        loss = l2loss + self._config["WeightPVBL2"] * pvbl2 + self._config["WeightPVBand"] * pvbloss
        kernelCurv = None
        if not curv is None:
            kernelCurv = torch.tensor(
                [[-1.0/16, 5.0/16, -1.0/16],
                 [ 5.0/16, -1.0,    5.0/16],
                 [-1.0/16, 5.0/16, -1.0/16]],
                dtype=REALTYPE, device=DEVICE
            )
            curvature = func.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
            losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
            loss += curv * losscurv

        # 初始梯度 G0
        loss.backward()
        G = params.grad.detach().clone()
        params.grad.zero_()

        # 初始速度 v0 = -G0
        v = -G.clone()

        # 最优记录
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = params.detach().clone()
        bestMask = mask.detach().clone()

        for idx in range(1, self._config["Iterations"]+1):
            # 时间步 Δt = λ_t / max(|v|)
            vmax = torch.max(torch.abs(v)).item() + 1e-12
            dt = self._config["StepSize"] / vmax

            # 参数更新：params_{i+1} = params_i + v_i * dt
            with torch.no_grad():
                params += v * dt

            # 前向计算
            mask, printedNom, printedMax, printedMin = self._levelset(params * self._filter + backup * (1.0 - self._filter))
            l2loss = func.mse_loss(printedNom, target, reduction="sum")
            pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
            pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            pvband = torch.sum((printedMax >= self._config["TargetDensity"]) != (printedMin >= self._config["TargetDensity"]))
            loss = l2loss + self._config["WeightPVBL2"] * pvbl2 + self._config["WeightPVBand"] * pvbloss
            if (not curv is None) and (kernelCurv is not None):
                curvature = func.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
                losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
                loss += curv * losscurv

            if verbose == 1:
                print(f"[Iter {idx}]: L2={l2loss.item():.0f}; PVBand={pvband.item():.0f}")

            # 最优保存
            if loss.item() < lossMin:
                lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
                bestParams = params.detach().clone()
                bestMask = mask.detach().clone()

            # 计算新梯度 G_{i+1}
            loss.backward()
            G_new = params.grad.detach().clone()
            params.grad.zero_()

            # 计算 SACG 系数 λ_i（PRP / FR 自适应）
            # PRP
            num_prp = torch.sum(G_new * (G_new - G))
            den = torch.sum(G * G) + 1e-12
            lambda_prp = num_prp / den
            # FR
            lambda_fr = torch.sum(G_new * G_new) / den
            # 自适应选择（简单条件示例：PRP>0用PRP，否则用FR）
            lambda_cg = lambda_prp if lambda_prp > 0 else lambda_fr

            # 更新速度 v_{i+1} = -G_{i+1} + λ_i * v_i
            v = -G_new + lambda_cg * v

            # 更新梯度历史
            G = G_new.clone()

            # 收敛判断
            if torch.max(torch.abs(v)).item() <= getattr(self._config, "Tol", 1e-4):
                break

        return l2Min, pvbMin, bestParams, bestMask


    # 🔧 添加带可视化的求解方法
    def solve_with_visualization(self, target, params, curv=None, verbose=0, visualize=False):
        """
        带可视化的求解方法
        """
        # Initialize
        backup = params
        params = params.clone().detach().requires_grad_(True)

        # Optimizer 
        opt = optim.Adam([params], lr=self._config["StepSize"])
        
        # 记录损失
        loss_history = []
        l2_history = []
        pvb_history = []
        
        # Optimization process
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None
        
        for idx in range(self._config["Iterations"]): 
            mask, printedNom, printedMax, printedMin = self._levelset(params * self._filter + backup * (1.0 - self._filter))
            l2loss = func.mse_loss(printedNom, target, reduction="sum")
            pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
            pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            pvband = torch.sum((printedMax >= self._config["TargetDensity"]) != (printedMin >= self._config["TargetDensity"]))
            loss = l2loss + self._config["WeightPVBL2"] * pvbl2 + self._config["WeightPVBand"] * pvbloss
            
            # 记录损失
            loss_history.append(loss.item())
            l2_history.append(l2loss.item())
            pvb_history.append(pvband.item())
            
            if not curv is None: 
                kernelCurv = torch.tensor([[-1.0/16, 5.0/16, -1.0/16], [5.0/16, -1.0, 5.0/16], [-1.0/16, 5.0/16, -1.0/16]], dtype=REALTYPE, device=DEVICE)
                curvature = func.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
                losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
                loss += curv * losscurv
                
            if verbose == 1: 
                print(f"[Iteration {idx:3d}]: L2={l2loss.item():8.0f}, PVB={pvband.item():4.0f}, Loss={loss.item():8.0f}")

            if bestParams is None or bestMask is None or loss.item() < lossMin: 
                lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
                bestParams = params.detach().clone()
                bestMask = mask.detach().clone()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # 如果需要可视化，绘制收敛曲线
        if visualize:
            plot_convergence_curve(loss_history, l2_history, pvb_history)
        
        return l2Min, pvbMin, bestParams, bestMask, loss_history, l2_history, pvb_history

    def compute_grad(self, target_norm, params, debug=False, debug_tag=""):
        device = params.device
        params = params.detach().requires_grad_(True)
        mask, printedNom, printedMax, printedMin = self._levelset(params * self._filter.to(device))
        target_norm = target_norm.to(dtype=REALTYPE, device=device)

        l2loss = func.mse_loss(printedNom, target_norm, reduction="sum")
        pvbl2 = func.mse_loss(printedMax, target_norm, reduction="sum") + func.mse_loss(printedMin, target_norm, reduction="sum")
        pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")

        # WeightPVBand = 0.0 if t < 20 else min((t - 20) / (50 - 20), 1.0)
        loss = l2loss + self._config["WeightPVBL2"] * pvbl2 + self._config["WeightPVBand"] * pvbloss

        loss.backward()
        grad = params.grad.detach()
        params.grad = None  # 可选：释放grad buffer

        if debug:
            grad_abs = grad.abs()
            print(
                f"[LevelSetDBG][compute_grad]{debug_tag} "
                f"loss={loss.item():.6f}, "
                f"grad_mean={grad_abs.mean().item():.6e}, "
                f"grad_max={grad_abs.max().item():.6e}, "
                f"grad_min={grad_abs.min().item():.6e}"
            )

        return {
            "gradient": grad,
            "loss": loss.item(),
            "l2loss": l2loss.item(),
            "pvbl2": pvbl2.item(),
            "pvbloss": pvbloss.item(),
        }

    def compute_grad_sacg(self, target_norm, params, t=0, tile_id=None):
        device = params.device
        params = params.detach().requires_grad_(True)
        mask, printedNom, printedMax, printedMin = self._levelset(params * self._filter)

        l2loss = func.mse_loss(printedNom, target_norm, reduction="sum")
        pvbl2  = func.mse_loss(printedMax, target_norm, reduction="sum") \
                + func.mse_loss(printedMin, target_norm, reduction="sum")
        pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")

        WeightPVBand = 0.0 if t < 20 else min((t - 20) / 30.0, 1.0)
        loss = l2loss + self._config["WeightPVBL2"] * pvbl2 + WeightPVBand * pvbloss

        loss.backward()
        grad = params.grad.detach()
        params.grad = None
        del mask, printedNom, printedMax, printedMin

        return {
            "gradient": grad,
            "loss": loss.item(),
            "l2loss": l2loss.item(),
            "pvbl2": pvbl2.item(),
            "pvbloss": pvbloss.item(),
            "WeightPVBand": WeightPVBand,
        }

    def apply_gradient(self, params, gradient, lr=1e-2, debug=False, debug_tag=""):
        old_params = params.detach().clone()
        update = -lr * gradient
        with torch.no_grad():
            params.add_(update)

        if debug:
            upd_abs = update.abs()
            delta_abs_mean = (params - old_params).abs().mean().item()
            print(
                f"[LevelSetDBG][apply_gradient]{debug_tag} "
                f"lr={lr:.6g}, "
                f"update_mean={upd_abs.mean().item():.6e}, "
                f"update_max={upd_abs.max().item():.6e}, "
                f"param_delta_mean={delta_abs_mean:.6e}, "
                f"param_mean={params.abs().mean().item():.6e}"
            )
        return params


    def apply_gradient_sacg(self, params, gradient, lr=1e-2, tile_id=None):
        device = params.device
        g = gradient.to(device).float()
        prev_grad = self._prev_grad_dict.get(tile_id)
        prev_dir  = self._prev_dir_dict.get(tile_id)
        beta_val = "N/A"

        if prev_grad is None or prev_dir is None:
            direction = -g
        else:
            pg = prev_grad.float()
            pd = prev_dir.float()
            den = torch.sum(pg * pg).clamp(min=1e-6)
            num = torch.sum(g * (g - pg))
            beta = torch.clamp(num / den, 0.0, 1.0)
            beta_val = f"{beta.item():.6g}"
            direction = -g + beta * pd
            direction = direction / (torch.norm(direction) + 1e-6)

        # float32累加，最后转回原dtype
        params32 = params.float()
        direction32 = direction.float()
        with torch.no_grad():
            new_params = (params32 + lr * direction32).to(params.dtype)

        # 打印步长和方向
        print(f"[Tile {tile_id}] update norm: {(lr * direction32).abs().mean().item():.6g}, direction norm: {direction32.abs().mean().item():.6g}, beta: {beta_val}")

        if tile_id is not None:
            self._prev_grad_dict[tile_id] = g.detach()
            self._prev_dir_dict[tile_id]  = direction.detach()

        return new_params

# 🔧 新增：单个OAS文件处理函数
def process_single_oas(oas_file, output_dir=None):
    """
    处理单个OAS文件
    """
    print(f"🔄 处理OAS文件: {os.path.basename(oas_file)}")
    
    # 读取OAS文件
    mask, result_info = read_oas_to_real_size_mask(oas_file, target_layers=None)
    if mask is None:
        raise ValueError(f"无法读取OAS文件: {oas_file}")
    
    # 归一化mask
    if mask.max() > 1.5:
        mask = mask / 255.0
    mask = mask.float()  # 替换掉 .astype(np.float32)
    print(f"  📊 Mask信息: 形状={mask.shape}, 覆盖率={np.sum(mask > 0.5)/mask.size*100:.2f}%")
    
    # 初始化配置和模型
    cfg = LevelSetCfg(str(DEFAULT_LEVELSET_CONFIG))
    litho = lithosim.LithoSim(str(DEFAULT_LITHO_CONFIG))
    solver = LevelSetILT(cfg, litho)
    
    # 使用LevelSetImageInit初始化
    # target, params = initializer.LevelSetImageInit().run(mask)
    
    # 更平滑的距离场初始化，减少角外凸
    target, params = initializer.LevelSetImageInitSmooth(blur_kernel=(5,5), blur_sigma=1.0, blur_times=2).run(mask)
    
    # 求解
    begin = time.time()
    l2, pvb, bestParams, bestMask = solver.solve(target, params, curv=None, verbose=1)
    runtime = time.time() - begin
    
    print(f"  ✅ 求解完成: L2={l2:.0f}, PVB={pvb:.0f}, 时间={runtime:.2f}s")
    
    # 保存结果
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "output")
    else:
        output_dir = str(_resolve_path(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(oas_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_levelset.png")
    cv2.imwrite(output_path, (bestMask * 255).detach().cpu().numpy())
    print(f"  💾 保存至: {output_path}")
    
    return l2, pvb, bestMask, runtime

# 🔧 修改后的serial函数：处理OAS文件而不是glp.Design
def serial_oas():
    """
    串行处理OAS文件
    """
    import glob
    
    # 🔧 查找OAS文件 - 使用绝对路径
    folder_path = str(ILT_DIR / "tiles_modified_noexpand1")
    oas_files = glob.glob(os.path.join(folder_path, "*.oas"))
    
    if not oas_files:
        print(f"❌ 在文件夹中没有找到OAS文件: {folder_path}")
        print(f"📁 请检查路径是否存在: {os.path.exists(folder_path)}")
        return
    
    oas_files.sort()
    print(f"📁 找到 {len(oas_files)} 个OAS文件")
    
    l2s = []
    pvbs = []
    runtimes = []
    
    cfg = LevelSetCfg(str(DEFAULT_LEVELSET_CONFIG))
    litho = lithosim.LithoSim(str(DEFAULT_LITHO_CONFIG))
    solver = LevelSetILT(cfg, litho)
    
    # 处理前几个文件作为测试
    test_files = oas_files[:3]  # 只处理前3个文件
    
    for idx, oas_file in enumerate(test_files, 1):
        print(f"\n[Testcase {idx}]: {os.path.basename(oas_file)}")
        
        # 读取OAS文件
        mask, result_info = read_oas_to_real_size_mask(oas_file, target_layers=None)
        if mask is None:
            print(f"  ❌ 跳过: 无法读取文件")
            continue
        
        # 归一化
        if mask.max() > 1.5:
            mask = mask / 255.0
        mask = mask.float()  # 替换掉 .astype(np.float32)
        
        # 初始化
        # target, params = initializer.LevelSetImageInit().run(mask)
        
        # 更平滑的距离场初始化，减少角外凸
        target, params = initializer.LevelSetImageInitSmooth(blur_kernel=(5,5), blur_sigma=1.0, blur_times=2).run(mask)
        
        # 求解
        begin = time.time()
        l2, pvb, bestParams, bestMask = solver.solve(target, params, curv=None, verbose=1)
        runtime = time.time() - begin
        
        # 保存结果
        output_path = str(PROJECT_ROOT / "tmp" / f"LevelSet_oas_{idx}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, (bestMask * 255).detach().cpu().numpy())
        
        print(f"  ✅ L2={l2:.0f}, PVB={pvb:.0f}, 时间={runtime:.2f}s")
        
        l2s.append(l2)
        pvbs.append(pvb)
        runtimes.append(runtime)
    
    if l2s:
        print(f"\n📊 总体结果: L2={np.mean(l2s):.0f}, PVB={np.mean(pvbs):.0f}, 平均时间={np.mean(runtimes):.2f}s")

# 在现有导入的基础上添加
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
# 🔧 修复中文字体显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def visualize_levelset_process(oas_file, mask, target, params, bestMask, bestParams, save_dir=None):
    """
    可视化LevelSet处理过程 - 修复中文显示
    """
    import matplotlib.pyplot as plt
    
    # 创建保存目录
    if save_dir is None:
        save_dir = str(PROJECT_ROOT / "visualizations")
    else:
        save_dir = str(_resolve_path(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(oas_file))[0]
    
    # 转换为numpy数组
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(params, torch.Tensor):
        params = params.detach().cpu().numpy()
    if isinstance(bestMask, torch.Tensor):
        bestMask = bestMask.detach().cpu().numpy()
    if isinstance(bestParams, torch.Tensor):
        bestParams = bestParams.detach().cpu().numpy()
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'LevelSet Optimization Process: {base_name}', fontsize=16, fontweight='bold')
    
    # 第一行：输入、目标、初始距离场
    # 原始OAS mask
    im1 = axes[0, 0].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original OAS Mask', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # 目标target
    im2 = axes[0, 1].imshow(target, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Initialization Target', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 初始距离场
    im3 = axes[0, 2].imshow(params, cmap='RdBu_r')
    axes[0, 2].set_title('Initial Distance Field', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    axes[0, 2].contour(params, levels=[0], colors='white', linewidths=2)  # 零水平集
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 第二行：优化结果、最终距离场、差异图
    # 最终优化结果
    im4 = axes[1, 0].imshow(bestMask, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Optimized Mask', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 最终距离场
    im5 = axes[1, 1].imshow(bestParams, cmap='RdBu_r')
    axes[1, 1].set_title('Final Distance Field', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    axes[1, 1].contour(bestParams, levels=[0], colors='white', linewidths=2)  # 零水平集
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 差异图
    diff = np.abs(bestMask - target)
    im6 = axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('Difference |Result - Target|', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # 添加统计信息（英文）
    total_pixels = target.size
    target_area = np.sum(target > 0.5)
    result_area = np.sum(bestMask > 0.5)
    diff_pixels = np.sum(diff > 0.1)
    
    info_text = f"""Statistics:
Target Area: {target_area:,} pixels ({target_area/total_pixels*100:.1f}%)
Result Area: {result_area:,} pixels ({result_area/total_pixels*100:.1f}%)
Area Diff: {result_area-target_area:+,} pixels
Diff Pixels: {diff_pixels:,} ({diff_pixels/total_pixels*100:.2f}%)
Distance Range: [{params.min():.1f}, {params.max():.1f}] -> [{bestParams.min():.1f}, {bestParams.max():.1f}]"""
    
    # 在图的右侧添加文本
    fig.text(0.02, 0.5, info_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, f"{base_name}_levelset_process.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Visualization saved to: {save_path}")
    
    # 🔧 不显示图片，只保存
    plt.close()  # 关闭图片，不显示
    
    return save_path

def plot_convergence_curve(losses, l2_losses, pvb_losses, save_dir=None, filename="convergence.png"):
    """
    绘制收敛曲线 - 修复中文显示
    """
    if not losses:
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 总损失
    axes[0].plot(losses, 'b-', linewidth=2)
    axes[0].set_title('Total Loss Convergence', fontweight='bold')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss Value')
    axes[0].grid(True, alpha=0.3)
    
    # L2损失
    if l2_losses:
        axes[1].plot(l2_losses, 'r-', linewidth=2)
        axes[1].set_title('L2 Loss Convergence', fontweight='bold')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('L2 Loss')
        axes[1].grid(True, alpha=0.3)
    
    # PVB损失
    if pvb_losses:
        axes[2].plot(pvb_losses, 'g-', linewidth=2)
        axes[2].set_title('PVB Loss Convergence', fontweight='bold')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('PVB Loss')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir is None:
        save_dir = str(PROJECT_ROOT / "visualizations")
    else:
        save_dir = str(_resolve_path(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  📈 Convergence curve saved to: {save_path}")
    
    # 🔧 不显示图片，只保存
    plt.close()  # 关闭图片，不显示
    return save_path

# 🔧 添加简洁的可视化函数
def quick_visualize(mask, target, bestMask, litho_nom, save_path="quick_comparison.png"):
    """
    快速对比可视化（无中文）
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # 原始mask
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 目标
    axes[1].imshow(target, cmap='gray')
    axes[1].set_title('Target')
    axes[1].axis('off')
    
    # 结果
    axes[2].imshow(bestMask, cmap='gray')
    axes[2].set_title('Mask')
    axes[2].axis('off')

    #光刻nominal
    axes[3].imshow(litho_nom, cmap='gray')
    axes[3].set_title('litho_nom')
    axes[3].axis('off')

    # 差异
    diff = np.abs(target - litho_nom)
    axes[4].imshow(diff, cmap='hot')
    axes[4].set_title('Difference')
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  🖼️  Quick comparison saved to: {save_path}")
    return save_path

# 🔧 修改处理函数，使用简洁可视化
def process_single_oas_with_visualization(oas_file, output_dir=None, visualize=True):
    """
    处理单个OAS文件并生成可视化
    """
    print(f"Processing OAS file: {os.path.basename(oas_file)}")
    
    # 读取OAS文件
    mask, result_info = read_oas_to_real_size_mask(oas_file, target_layers=None)
    if mask is None:
        raise ValueError(f"Cannot read OAS file: {oas_file}")
    
    # 归一化mask
    if mask.max() > 1.5:
        mask = mask / 255.0
    mask = mask.float()  # 替换掉 .astype(np.float32)
    
    print(f"  📊 Mask info: shape={mask.shape}, coverage={np.sum(mask > 0.5)/mask.size*100:.2f}%")
    
    # 初始化配置和模型
    cfg = LevelSetCfg(str(DEFAULT_LEVELSET_CONFIG))
    litho = lithosim.LithoSim(str(DEFAULT_LITHO_CONFIG))
    solver = LevelSetILT(cfg, litho)
    
    # 使用LevelSetImageInit初始化
    target, params = initializer.LevelSetImageInit().run(mask)
    initial_params = params.clone()  # 保存初始参数用于可视化
    
    print(f"  🎯 Target stats: pixels={torch.sum(target).item():,.0f}, coverage={torch.sum(target)/target.numel()*100:.2f}%")
    print(f"  🔧 Initial distance field: range=[{params.min():.2f}, {params.max():.2f}]")
    
    # 求解
    begin = time.time()
    l2, pvb, bestParams, bestMask, loss_history, l2_history, pvb_history = solver.solve_with_visualization(
        target, params, curv=None, verbose=1, visualize=False  # 🔧 关闭自动显示
    )
    runtime = time.time() - begin
    
    print(f"  ✅ Optimization completed: L2={l2:.0f}, PVB={pvb:.0f}, time={runtime:.2f}s")

    with torch.no_grad():
        litho_input = bestMask if isinstance(bestMask, torch.Tensor) else torch.tensor(bestMask, dtype=torch.float32)
        litho_nom_preview, _, _ = litho(litho_input)
    
    # 生成可视化
    if visualize:
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / "output")
        else:
            output_dir = str(_resolve_path(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(oas_file))[0]
        
        # 详细可视化
        visualize_levelset_process(oas_file, mask, target, initial_params, bestMask, bestParams, output_dir)
        
        # 收敛曲线
        plot_convergence_curve(loss_history, l2_history, pvb_history, output_dir, f"{base_name}_convergence.png")
        
        # 快速对比
        quick_path = os.path.join(output_dir, f"{base_name}_quick_comparison.png")
        quick_visualize(mask, target.detach().cpu().numpy(), bestMask.detach().cpu().numpy(), litho_nom_preview.detach().cpu().numpy(), quick_path)
    
    # 保存结果
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "output")
    else:
        output_dir = str(_resolve_path(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(oas_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_levelset.png")
    cv2.imwrite(output_path, (bestMask * 255).detach().cpu().numpy())
    print(f"  💾 Result saved to: {output_path}")
    
    # ====== 新增：再进入一次光刻并保存光刻结果 ======
    with torch.no_grad():
        if isinstance(bestMask, torch.Tensor):
            mask_tensor = bestMask
        else:
            mask_tensor = torch.tensor(bestMask, dtype=torch.float32)
        # 光刻仿真
        printedNom, printedMax, printedMin = litho(mask_tensor)
        # 保存光刻仿真结果
        litho_nom_path = os.path.join(output_dir, f"{base_name}_litho_nom.png")
        litho_max_path = os.path.join(output_dir, f"{base_name}_litho_max.png")
        litho_min_path = os.path.join(output_dir, f"{base_name}_litho_min.png")
        cv2.imwrite(litho_nom_path, (printedNom.detach().cpu().numpy() * 255).astype('uint8'))
        cv2.imwrite(litho_max_path, (printedMax.detach().cpu().numpy() * 255).astype('uint8'))
        cv2.imwrite(litho_min_path, (printedMin.detach().cpu().numpy() * 255).astype('uint8'))
        print(f"  💾 Litho仿真结果已保存: {litho_nom_path}, {litho_max_path}, {litho_min_path}")
    # ====== 新增结束 ======
    return l2, pvb, bestMask, runtime

# 🔧 或者简化版本，直接使用原来的solve方法
def process_single_oas_simple_visualization(oas_file, output_dir=None, visualize=True):
    """
    处理单个OAS文件并生成可视化 - 简化版本
    """
    print(f"Processing OAS file: {os.path.basename(oas_file)}")
    
    # 读取OAS文件
    mask, result_info = read_oas_to_real_size_mask(oas_file, target_layers=None)
    if mask is None:
        raise ValueError(f"Cannot read OAS file: {oas_file}")
    
    # 归一化mask
    if mask.max() > 1.5:
        mask = mask / 255.0
    mask = mask.float()  # 替换掉 .astype(np.float32)
    
    print(f"  📊 Mask info: shape={mask.shape}, coverage={np.sum(mask > 0.5)/mask.size*100:.2f}%")
    
    # 初始化配置和模型
    cfg = LevelSetCfg(str(DEFAULT_LEVELSET_CONFIG))
    litho = lithosim.LithoSim(str(DEFAULT_LITHO_CONFIG))
    solver = LevelSetILT(cfg, litho)
    
    # 使用LevelSetImageInit初始化
    target, params = initializer.LevelSetImageInit().run(mask)
    initial_params = params.clone()  # 保存初始参数用于可视化
    
    print(f"  🎯 Target stats: pixels={torch.sum(target).item():,.0f}, coverage={torch.sum(target)/target.numel()*100:.2f}%")
    print(f"  🔧 Initial distance field: range=[{params.min():.2f}, {params.max():.2f}]")
    
    # 🔧 使用原来的solve方法
    begin = time.time()
    l2, pvb, bestParams, bestMask = solver.solve_sacg(target, params, curv=None, verbose=1)
    runtime = time.time() - begin
    
    print(f"  ✅ Optimization completed: L2={l2:.0f}, PVB={pvb:.0f}, time={runtime:.2f}s")
    
    
    # 保存结果
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "output")
    else:
        output_dir = str(_resolve_path(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(oas_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_levelset.png")
    cv2.imwrite(output_path, (bestMask * 255).detach().cpu().numpy())
    print(f"  💾 Result saved to: {output_path}")
    
    # ====== 新增：再进入一次光刻并保存光刻结果 ======
    with torch.no_grad():
        if isinstance(bestMask, torch.Tensor):
            mask_tensor = bestMask
        else:
            mask_tensor = torch.tensor(bestMask, dtype=torch.float32)
        # 光刻仿真
        printedNom, printedMax, printedMin = litho(mask_tensor)
        # 保存光刻仿真结果
        litho_nom_path = os.path.join(output_dir, f"{base_name}_litho_nom.png")
        litho_max_path = os.path.join(output_dir, f"{base_name}_litho_max.png")
        litho_min_path = os.path.join(output_dir, f"{base_name}_litho_min.png")
        cv2.imwrite(litho_nom_path, (printedNom.detach().cpu().numpy() * 255).astype('uint8'))
        cv2.imwrite(litho_max_path, (printedMax.detach().cpu().numpy() * 255).astype('uint8'))
        cv2.imwrite(litho_min_path, (printedMin.detach().cpu().numpy() * 255).astype('uint8'))
        print(f"  💾 Litho仿真结果已保存: {litho_nom_path}, {litho_max_path}, {litho_min_path}")
    # ====== 新增结束 ======
    # 生成可视化
    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(oas_file))[0]
        
        # 详细可视化
        visualize_levelset_process(oas_file, mask, target, initial_params, bestMask, bestParams, output_dir)
        
        # 快速对比
        quick_path = os.path.join(output_dir, f"{base_name}_quick_comparison.png")
        litho_nom_np = printedNom.detach().cpu().numpy()
        quick_visualize(mask, target.detach().cpu().numpy(), bestMask.detach().cpu().numpy(), litho_nom_np, quick_path)

    return l2, pvb, bestMask, runtime

# 🔧 修改serial_oas函数，使用简化版本
def serial_oas_with_visualization():
    """
    串行处理OAS文件并生成可视化
    """
    import glob
    
    # 🔧 查找OAS文件 - 使用绝对路径
    folder_path = str(ILT_DIR / "testtiles")
    oas_files = glob.glob(os.path.join(folder_path, "*.oas"))
    
    if not oas_files:
        print(f"❌ 在文件夹中没有找到OAS文件: {folder_path}")
        print(f"📁 请检查路径是否存在: {os.path.exists(folder_path)}")
        return
    
    oas_files.sort()
    print(f"📁 找到 {len(oas_files)} 个OAS文件")
    
    # 只处理第一个文件，生成详细可视化
    test_file = oas_files[13]
    print(f"\n🎯 处理文件: {os.path.basename(test_file)}")
    
    try:
        # 🔧 使用简化版本
        l2, pvb, bestMask, runtime = process_single_oas_simple_visualization(
            test_file, 
            output_dir=str(PROJECT_ROOT / "visualizations1"),
            visualize=True
        )
        print(f"\n✅ 处理完成！请查看 '{str(PROJECT_ROOT / 'visualizations1')}' 文件夹中的可视化结果")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()




class AdaptiveLevelSetILT(LevelSetILT):
    """
    自适应尺寸的LevelSet求解器，自动选择配置和模型，支持大于1024的参数张量
    """
    def __init__(self, base_config=None, lithosim_config=None, device=DEVICE):
        if base_config is None:
            base_config = str(DEFAULT_LEVELSET_CONFIG)
        else:
            base_config = str(_resolve_path(base_config))
        if lithosim_config is None:
            lithosim_config = str(DEFAULT_LITHO_CONFIG)
        else:
            lithosim_config = str(_resolve_path(lithosim_config))

        cfg = LevelSetCfg(base_config)
        litho = lithosim.LithoSim(lithosim_config)
        super().__init__(cfg, litho, device=device)
        self.base_config_path = base_config
        self.lithosim_config_path = lithosim_config

    def adapt_config(self, params_tensor):
        sizeY, sizeX = params_tensor.shape[-2], params_tensor.shape[-1]
        cfg_dict = common.parseConfig(self.base_config_path)
        cfg_dict["TileSizeX"] = sizeX
        cfg_dict["TileSizeY"] = sizeY
        cfg_dict["ILTSizeX"] = sizeX
        cfg_dict["ILTSizeY"] = sizeY
        cfg_dict["OffsetX"] = 0
        cfg_dict["OffsetY"] = 0
        self._config = cfg_dict
        self._filter = torch.ones([sizeY, sizeX], dtype=REALTYPE, device=self._device)  # 注意顺序

    def params_to_mask(self, params_tensor):
        self.adapt_config(params_tensor)  # 保证每次都适配
        device = params_tensor.device
        filter_tensor = self._filter.to(device)
        mask, _, _, _ = self._levelset(params_tensor * filter_tensor)
        return mask

def create_adaptive_solver(base_config=None, lithosim_config=None, device=DEVICE):
    return AdaptiveLevelSetILT(base_config, lithosim_config, device)

    
