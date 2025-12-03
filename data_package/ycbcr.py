import torch
import torch.nn as nn
import math

class ContrastPreservedChromaEnhancement(nn.Module):
    def __init__(self, chroma_weight: float = 1.2):
        """
        初始化对比度保持色度增强模块。
        
        Args:
            chroma_weight (float): 色度增强权重 (cw). 
                                   例如 1.2 表示增强 20%[cite: 16].
        """
        super().__init__()
        self.chroma_weight = chroma_weight
        
        # 定义 ITU-R BT.709 的 RGB 转 YCbCr 转换矩阵 (近似值)
        # Y range: 0-1, Cb/Cr range: -0.5 to 0.5 (normalized)
        self.register_buffer('rgb_to_ycbcr_kernel', torch.tensor([
            [0.2126, 0.7152, 0.0722],
            [-0.1146, -0.3854, 0.5000],
            [0.5000, -0.4542, -0.0458]
        ]).view(3, 3, 1, 1))

        self.register_buffer('ycbcr_to_rgb_kernel', torch.tensor([
            [1.0, 0.0, 1.5748],
            [1.0, -0.1873, -0.4681],
            [1.0, 1.8556, 0.0]
        ]).view(3, 3, 1, 1))

        # --- 模型参数 (a1 - a5) ---
        # 注意：论文并未直接给出这些数值表格，只说明了需要针对6个主色相进行计算 [cite: 103, 129]。
        # 这里的数值仅为占位符 (Placeholder)，实际使用时需根据具体显示设备校准或论文补充材料填入。
        # 格式: { '色相角(度)': [a1, a2, a3, a4, a5] }
        # 假设色相角顺序: Red(0), Yellow(60), Green(120), Cyan(180), Blue(240), Magenta(300)
        self.hue_angles = torch.tensor([0.0, 60.0, 120.0, 180.0, 240.0, 300.0])
        self.hue_params = torch.tensor([
            # a1,   a2,   a3,   a4,   a5  (以下均为随机示例值，非论文真实数据)
            [0.5,  1.2,  0.5,  0.1,  0.8], # Red
            [0.4,  1.1,  0.4,  0.1,  0.8], # Yellow
            [0.3,  1.0,  0.3,  0.1,  0.9], # Green
            [0.3,  1.0,  0.3,  0.1,  0.9], # Cyan
            [0.4,  1.1,  0.4,  0.1,  0.8], # Blue
            [0.5,  1.2,  0.5,  0.1,  0.8], # Magenta
        ])

    def rgb_to_ycbcr(self, img_rgb):
        """将 RGB (0-1) 转换为 YCbCr (Y:0-1, CbCr: -0.5 to 0.5)"""
        return nn.functional.conv2d(img_rgb, self.rgb_to_ycbcr_kernel)

    def ycbcr_to_rgb(self, img_ycbcr):
        """将 YCbCr 转换回 RGB"""
        return nn.functional.conv2d(img_ycbcr, self.ycbcr_to_rgb_kernel)

    def get_interpolated_params(self, hue_angle_deg):
        """
        根据像素的色相角，对相邻主色的参数进行线性插值 。
        
        Args:
            hue_angle_deg (Tensor): (B, 1, H, W) 色相角度 0-360
        Returns:
            params (Tensor): (B, 5, H, W) 插值后的 a1-a5 参数
        """
        # 确保角度在 0-360 范围内
        hue = hue_angle_deg % 360.0
        
        # 找到各个像素对应的区间索引 (0-5)
        # 例如: 45度 -> 位于 index 0 (0度) 和 index 1 (60度) 之间
        idx_lower = (hue / 60.0).floor().long() % 6
        idx_upper = (idx_lower + 1) % 6
        
        # 计算插值权重 alpha
        hue_lower = idx_lower.float() * 60.0
        alpha = (hue - hue_lower) / 60.0
        # 处理跨越 360/0 度的情况
        alpha = torch.where(hue < hue_lower, (hue + 360.0 - hue_lower) / 60.0, alpha)
        
        # 获取对应参数 (B, H, W, 5) -> Permute to (B, 5, H, W)
        # 注意：这里为了高效，使用了 index_select 的逻辑，但在 4D Tensor 上需要 gather
        
        # 将参数移动到与输入相同的设备
        params_device = self.hue_params.to(hue.device)
        
        # Gather logic implementation
        p_lower = params_device[idx_lower] # (B, 1, H, W, 5) if indexed directly, need careful shape handling
        
        # 简化维度处理: Flatten -> Index -> Reshape
        b, c, h, w = hue.shape
        flat_idx_lower = idx_lower.view(-1)
        flat_idx_upper = idx_upper.view(-1)
        
        p_lower = params_device.index_select(0, flat_idx_lower).view(b, h, w, 5).permute(0, 3, 1, 2)
        p_upper = params_device.index_select(0, flat_idx_upper).view(b, h, w, 5).permute(0, 3, 1, 2)
        
        # 线性插值
        current_params = p_lower * (1 - alpha) + p_upper * alpha
        return current_params

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入图像 RGB, 范围 [0, 1], 形状 (B, 3, H, W)
        Returns:
            Tensor: 增强后的 RGB 图像
        """
        # 1. 转换到 YCbCr 空间
        ycbcr = self.rgb_to_ycbcr(x)
        y_in = ycbcr[:, 0:1, :, :]
        cb_in = ycbcr[:, 1:2, :, :]
        cr_in = ycbcr[:, 2:3, :, :]

        # 2. 计算几何属性 (Page 1, Equation definition in Sec II)
        # YCC_chroma = sqrt(Cb^2 + Cr^2) 
        # 注意：原文使用 0-255范围，这里我们需要对数值进行相应缩放以匹配公式量级
        # 假设输入 Y 是 0-1，则乘以 255 恢复到论文尺度
        y_255 = y_in * 255.0
        cb_255 = cb_in * 255.0
        cr_255 = cr_in * 255.0
        
        chroma = torch.sqrt(cb_255**2 + cr_255**2)
        
        # 计算 Hue (角度)
        # atan2 返回弧度 -pi 到 pi，转换为度数 0-360
        hue_rad = torch.atan2(cr_255, cb_255) 
        hue_deg = (torch.degrees(hue_rad) + 360.0) % 360.0

        # 3. 获取插值后的参数 a1-a5
        # params: (B, 5, H, W)
        params = self.get_interpolated_params(hue_deg)
        a1 = params[:, 0:1, :, :]
        a2 = params[:, 1:2, :, :]
        a3 = params[:, 2:3, :, :]
        a4 = params[:, 3:4, :, :]
        a5 = params[:, 4:5, :, :]

        # 4. 计算 Delta Luma (核心公式 Eq. 1) 
        # 公式: (Chroma^a1) * {a2 * (ln(luma/255) + 0.1) + a3} * a4 * (cw^2 + a5*cw + a5 - 1)
        cw = self.chroma_weight
        
        # 防止 log(0)
        luma_log_term = torch.log((y_255 / 255.0) + 1e-6) + 0.1
        
        term1 = torch.pow(chroma, a1)
        term2 = a2 * luma_log_term + a3
        term3 = a4 * (cw**2 + a5 * cw + a5 - 1.0)
        
        delta_luma = term1 * term2 * term3 # 此时 delta_luma 是 0-255 尺度

        # 5. 应用增强与补偿 [cite: 116, 117]
        # 色度增强: 直接乘以权重
        cb_out = cb_in * cw
        cr_out = cr_in * cw
        
        # 亮度补偿: Y_out = Y_in + delta_luma (需归一化回 0-1)
        y_out = y_in + (delta_luma / 255.0)
        
        # 截断到合法范围
        y_out = torch.clamp(y_out, 0.0, 1.0)

        # 6. 重建图像
        ycbcr_out = torch.cat([y_out, cb_out, cr_out], dim=1)
        rgb_out = self.ycbcr_to_rgb(ycbcr_out)
        
        return torch.clamp(rgb_out, 0.0, 1.0)

# --- 使用示例 ---
if __name__ == "__main__":
    # 模拟一张图片 (Batch=1, Channels=3, Height=256, Width=256)
    input_image = torch.rand(1, 3, 256, 256)
    
    # 实例化模型，设定增强倍数为 1.3 (30% 增强)
    model = ContrastPreservedChromaEnhancement(chroma_weight=1.3)
    
    # 前向传播
    output_image = model(input_image)
    
    print(f"Input shape: {input_image.shape}")
    print(f"Output shape: {output_image.shape}")
    print(f"Enhancement applied with weight: {model.chroma_weight}")