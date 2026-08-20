import os
# 🔧 在所有其他导入之前添加这行，解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

# Switch the _computeImage final reduction backend via env var (P0.2).
#   FUILT_USE_TRITON_ABSPOW=1  → fused Triton reduction (1 launch instead of 4)
#   FUILT_USE_TRITON_ABSPOW=0  → original PyTorch eager (abs + pow + mul + sum)
#
# Mathematically equivalent; the Triton port saves 3 launches + 3 full passes
# over the [K, H, W] complex tensor.
_USE_TRITON_ABSPOW = bool(int(os.environ.get("FUILT_USE_TRITON_ABSPOW", "0")))


def _abs_pow_scale_sum(tmp, scale, dim):
    """Pick backend: Triton-fused if env var set, else PyTorch eager.

    `scale` arrives as broadcastable shape (e.g. [K,1,1] or [1,K,1,1]).
    For the Triton path we squeeze to 1D [K]; for the eager path we use
    the broadcastable form directly.
    """
    if _USE_TRITON_ABSPOW and tmp.dim() in (3, 4):
        # Lazy import so users not using this path don't pay triton JIT cost
        from .triton_abs_pow_scaled import abs_pow_scale_sum_triton
        scale_1d = scale.view(-1)  # [K]
        return abs_pow_scale_sum_triton(tmp, scale_1d)
    return torch.sum(scale * torch.pow(torch.abs(tmp), 2), dim=dim)


try:
    from . import settings as _settings
    from . import utils as common
except ImportError:
    import settings as _settings
    import utils as common

REALTYPE = _settings.REALTYPE
COMPLEXTYPE = _settings.COMPLEXTYPE
DEVICE = _settings.DEVICE



class Kernel:
    def __init__(self, basedir=None, defocus=False, conjuncture=False, combo=False, device=DEVICE):
        base_dir = Path(__file__).resolve().parent.parent

        if basedir is None:
            self._basedir = base_dir / "kernel"
        else:
            basedir_path = Path(basedir)
            if basedir_path.is_absolute():
                self._basedir = basedir_path
            else:
                self._basedir = (base_dir / basedir_path).resolve()
        self._defocus = defocus
        self._conjuncture = conjuncture
        self._combo = combo
        self._device = device

        self._kernels = torch.load(self._kernel_file(), map_location=device).permute(2, 0, 1)
        self._scales = torch.load(self._scale_file(), map_location=device)

        self._knx, self._kny = self._kernels.shape[:2]

    @property
    def kernels(self): 
        return self._kernels
        
    @property
    def scales(self): 
        return self._scales

    def _kernel_file(self):
        filename = ""
        if self._defocus:
            filename = "defocus"
        else:
            filename = "focus"
        if self._conjuncture:
            filename = "ct_" + filename
        if self._combo:
            filename = "combo_" + filename
        return str(self._basedir / "kernels" / f"{filename}.pt")

    def _scale_file(self):
        base = self._basedir / "scales"
        if self._combo:
            return str(base / "combo.pt")
        else:
            if self._defocus:
                return str(base / "defocus.pt")
            else:
                return str(base / "focus.pt")

def _maskFloat(mask, dose):
    return (dose * mask).to(COMPLEXTYPE)

def _kernelMult(kernel, maskFFT, kernelNum):
    # kernel: [24, 35, 35]
    knx, kny = kernel.shape[-2:]
    knxh, knyh = knx // 2, kny // 2
    output = None
    if kernel.device != maskFFT.device: 
        kernel = kernel.to(maskFFT.device)
    if maskFFT.shape[0] == 1: 
        output = torch.zeros([kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :knxh+1, :knyh+1] = maskFFT[:, :knxh+1, :knyh+1] * kernel[:kernelNum, -(knxh+1):, -(knyh+1):]
        output[:, :knxh+1, -knyh:] = maskFFT[:, :knxh+1, -knyh:] * kernel[:kernelNum, -(knxh+1):, :knyh]
        output[:, -knxh:, :knyh+1] = maskFFT[:, -knxh:, :knyh+1] * kernel[:kernelNum, :knxh, -(knyh+1):]
        output[:, -knxh:, -knyh:] = maskFFT[:, -knxh:, -knyh:] * kernel[:kernelNum, :knxh, :knyh]
    else: 
        maskFFT = torch.unsqueeze(maskFFT, 1)
        output = torch.zeros([maskFFT.shape[0], kernelNum, maskFFT.shape[-2], maskFFT.shape[-1]], dtype=maskFFT.dtype, device=maskFFT.device)
        output[:, :, :knxh+1, :knyh+1] = maskFFT[:, :, :knxh+1, :knyh+1] * kernel[None, :kernelNum, -(knxh+1):, -(knyh+1):]
        output[:, :, :knxh+1, -knyh:] = maskFFT[:, :, :knxh+1, -knyh:] * kernel[None, :kernelNum, -(knxh+1):, :knyh]
        output[:, :, -knxh:, :knyh+1] = maskFFT[:, :, -knxh:, :knyh+1] * kernel[None, :kernelNum, :knxh, -(knyh+1):]
        output[:, :, -knxh:, -knyh:] = maskFFT[:, :, -knxh:, -knyh:] * kernel[None, :kernelNum, :knxh, :knyh]
    return output

def _computeImage(cmask, kernel, scale, kernelNum):
    #频域卷积得到光强输出
    # cmask: [2048, 2048], kernel: [24, 35, 35], scale: [24]
    if scale.device != cmask.device: 
        scale = scale.to(cmask.device)
    if len(cmask.shape) == 2: 
        cmask = torch.unsqueeze(cmask, 0)
    cmask_fft = torch.fft.fft2(cmask, norm="forward")
    tmp = _kernelMult(kernel, cmask_fft, kernelNum)
    tmp = torch.fft.ifft2(tmp, norm="forward")
    if len(tmp.shape) == 3:
        if kernelNum == 1:
            return tmp[0]
        scale = scale[:kernelNum].unsqueeze(1).unsqueeze(2)
        return _abs_pow_scale_sum(tmp, scale.view(-1, 1, 1), dim=0)
    assert len(tmp.shape) == 4
    if kernelNum == 1:
        return tmp[:, 0]
    scale = scale[None, :kernelNum, None, None]
    return _abs_pow_scale_sum(tmp, scale.view(1, -1, 1, 1), dim=1)

#复数掩模，剂量，核，缩放，核数量
def _convMatrix(cmask, dose, kernel, scale, kernelNum): 
    image = _computeImage(cmask, kernel, scale, kernelNum)
    return image
#实数掩模，剂量，核，缩放，核数量
def _convMask(mask, dose, kernel, scale, kernelNum): 
    cmask = _maskFloat(mask, dose)
    image = _computeImage(cmask, kernel, scale, kernelNum)
    return image

class _LithoSim(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad): 
        ctx.saved = (mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad)
        return _convMask(mask, dose, kernel, scale, kernelNum)
        #实数掩膜 → _convMask → 光强
    @staticmethod
    def backward(ctx, *grad_outputs): 
        grad = grad_outputs[0]
        (mask, dose, kernel, scale, kernelNum, kernelGradCT, scaleGradCT, kernelNumGradCT, kernelGrad, scaleGrad, kernelNumGrad) = ctx.saved
        # 路径1 (CT核→正向核)
        cpx0 = torch.mul(_convMask(mask, dose, kernelGradCT, scaleGradCT, kernelNumGradCT), grad)
        #kernelGradCT共轭转置专用
        cpx1 = _convMatrix(cpx0, dose, kernelGrad, scaleGrad, kernelNumGrad)
        # 路径2 (正向核→CT核)
        cpx2 = torch.mul(_convMask(mask, dose, kernelGrad, scaleGrad, kernelNumGrad), grad)
        cpx3 = _convMatrix(cpx2, dose, kernelGradCT, scaleGradCT, kernelNumGradCT)
        cpx4 = cpx1 + cpx3#梯度合并
        return cpx4.real, None, None, None, None, None, None, None, None, None, None
        #只计算mask的梯度，其他参数不计算梯度
class LithoSim(nn.Module): # Mask -> Aerial -> Printed
    def __init__(self, config): 
        super(LithoSim, self).__init__()
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        elif isinstance(config, str): 
            self._config = common.parseConfig(config)
        required = ["KernelDir", "KernelNum", "TargetDensity", "PrintThresh", "PrintSteepness", "DoseMax", "DoseMin", "DoseNom"]
        for key in required: 
            assert key in self._config, f"[LithoSim]: Cannot find the config {key}."
        intfields = ["KernelNum", ]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "PrintThresh", "PrintSteepness", "DoseMax", "DoseMin", "DoseNom"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])
        # Read the kernels
        self._kernels = {"focus": Kernel(self._config["KernelDir"]), 
                         "defocus": Kernel(self._config["KernelDir"], defocus=True),
                         "CT focus": Kernel(self._config["KernelDir"], conjuncture=True),
                         "CT defocus": Kernel(self._config["KernelDir"], defocus=True, conjuncture=True),
                         "combo focus": Kernel(self._config["KernelDir"], combo=True),
                         "combo defocus": Kernel(self._config["KernelDir"], defocus=True, combo=True),
                         "combo CT focus": Kernel(self._config["KernelDir"], conjuncture=True, combo=True),
                         "combo CT defocus": Kernel(self._config["KernelDir"], defocus=True, conjuncture=True, combo=True)}
        # for name, kernel in self._kernels.items(): 
        #     kernel._kernels = nn.Parameter(kernel.kernels)
        #     kernel._scales = nn.Parameter(kernel.scales)
        #     self.register_parameter(f"{name} K", kernel.kernels)
        #     self.register_parameter(f"{name} S", kernel.scales)

    def forward(self, mask): 
        aerialNom = _LithoSim.apply(mask, self._config["DoseNom"], 
                                    self._kernels["focus"].kernels, self._kernels["focus"].scales, self._config["KernelNum"], 
                                    self._kernels["combo CT focus"].kernels, self._kernels["combo CT focus"].scales, 1, 
                                    self._kernels["combo focus"].kernels, self._kernels["combo focus"].scales, 1)
        aerialMax = _LithoSim.apply(mask, self._config["DoseMax"], 
                                    self._kernels["focus"].kernels, self._kernels["focus"].scales, self._config["KernelNum"], 
                                    self._kernels["combo CT focus"].kernels, self._kernels["combo CT focus"].scales, 1, 
                                    self._kernels["combo focus"].kernels, self._kernels["combo focus"].scales, 1)
        aerialMin = _LithoSim.apply(mask, self._config["DoseMin"], 
                                    self._kernels["defocus"].kernels, self._kernels["defocus"].scales, self._config["KernelNum"], 
                                    self._kernels["combo CT defocus"].kernels, self._kernels["combo CT defocus"].scales, 1, 
                                    self._kernels["combo defocus"].kernels, self._kernels["combo defocus"].scales, 1)
        printedNom = torch.sigmoid(self._config["PrintSteepness"] * (aerialNom - self._config["TargetDensity"]))
        printedMax = torch.sigmoid(self._config["PrintSteepness"] * (aerialMax - self._config["TargetDensity"]))
        printedMin = torch.sigmoid(self._config["PrintSteepness"] * (aerialMin - self._config["TargetDensity"]))
        return printedNom, printedMax, printedMin

