import numpy as np
import torch
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn.functional as F

def self_attention(query, key, value, mask=None):
    """
    计算自注意力（Self-Attention）

    Args:
        query (torch.Tensor): Query张量，形状为 (batch_size, seq_len, d_model)
        key (torch.Tensor): Key张量，形状为 (batch_size, seq_len, d_model)
        value (torch.Tensor): Value张量，形状为 (batch_size, seq_len, d_model)
        mask (torch.Tensor, optional): 可选的掩码，用于屏蔽掉不需要计算的部分，形状为 (batch_size, seq_len)

    Returns:
        torch.Tensor: Attention后的输出，形状为 (batch_size, seq_len, d_model)
    """
    d_k = query.size(-1)  # 获取Key的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)

    return output, attention_weights

def motion_appearance_attention(position_velocity, appearance_feature, attention_dim=128):
    """
    计算位置、速度和外观特征之间的自注意力加权融合

    Args:
        position_velocity (torch.Tensor): 运动模型的张量，包含位置和速度，形状为 (batch_size, seq_len, 2)
        appearance_feature (torch.Tensor): 外观特征的张量，形状为 (batch_size, seq_len, d_appearance)
        attention_dim (int, optional): Attention的维度，默认为128

    Returns:
        torch.Tensor: 融合后的特征表示，形状为 (batch_size, seq_len, d_model)
    """
    # 为位置和速度创建Q、K、V
    batch_size, seq_len, _ = position_velocity.size()

    # 假设外观特征维度是d_appearance，设定attention_dim来统一投影空间
    Q_motion = position_velocity  # 位置和速度作为Q
    K_motion = position_velocity  # 位置和速度作为K
    V_motion = position_velocity  # 位置和速度作为V

    Q_appearance = appearance_feature  # 外观特征作为Q
    K_appearance = appearance_feature  # 外观特征作为K
    V_appearance = appearance_feature  # 外观特征作为V

    # 计算运动模型部分的注意力
    motion_attention_output, motion_attention_weights = self_attention(Q_motion, K_motion, V_motion)

    # 计算外观特征部分的注意力
    appearance_attention_output, appearance_attention_weights = self_attention(Q_appearance, K_appearance, V_appearance)

    # 融合运动模型与外观特征的注意力输出
    combined_output = motion_attention_output + appearance_attention_output

    return combined_output
