import numpy as np

def wrap_to_pi(angle):
    """
    将角度归一化到 [-π, π) 区间。
    
    在航向控制中，角度差可能超出 ±π（例如从 3.1 rad 到 -3.1 rad），
    直接相减会得到约 6.2 rad 的误差，而实际转向仅需 ~0.08 rad。
    此函数确保角度差始终取最短路径。
    
    参数:
        angle: 输入角度（弧度），可以是标量或 numpy 数组
    
    返回:
        归一化到 [-π, π) 的角度
    
    示例:
        wrap_to_pi(3.5)   → -2.783...  (3.5 - 2π)
        wrap_to_pi(-3.5)  → 2.783...   (-3.5 + 2π)
        wrap_to_pi(0.5)   → 0.5        (无变化)
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
