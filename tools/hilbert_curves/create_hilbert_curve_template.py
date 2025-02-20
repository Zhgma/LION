"""edited:
~change N into current at #flag1
pep-8, variable name
assert
~deepcopy the input tensor "xy" (otherwise xy will be cleared!!)
~check: print(xy.sum() == 0) in the function "convert"
draw a curve
"""

import torch
import matplotlib.pyplot as plt

def hilbert_curve_to_tensor(orders):
    """
    生成指定阶数n的Hilbert曲线，并将其表示为张量形式。
    
    参数:
        n (int): 曲线的阶数，决定曲线的大小为2^n x 2^n。
    
    返回:
        torch.Tensor: 一个2^n x 2^n的张量，每个位置的值表示曲线经过的顺序。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_order = max(orders)
    # 生成Hilbert曲线的坐标
    hilbert_curve(max_order, orders, device)

def hilbert_curve(order, orders, device):
    """
    生成指定阶数n的Hilbert曲线的坐标列表。
    
    参数:
        n (int): 曲线的阶数，决定曲线的大小为2^n x 2^n。
    
    返回:
        list: 按顺序排列的坐标列表，每个元素为(x, y)形式的元组。
    """
    if order == 0:
        return torch.tensor([[0]], device=device)
    
    s = 2 ** (order - 1)
    curve = torch.zeros(2 * s, 2 * s, device=device)
    # 递归生成n-1阶的曲线
    prev_curve = hilbert_curve(order - 1, orders, device=device)
    curve[:s, :s] = prev_curve.clone().detach().permute(1, 0)
    curve[:s, s:] = prev_curve.clone().detach() + s * s
    curve[s:, s:] = prev_curve.clone().detach() + 2 * s * s
    curve[s:, :s] = prev_curve.clone().detach().permute(1, 0).flip(0, 1) + 3 * s * s
    
    if order in orders:
        curve_flat = curve.view(-1)
        torch.save(curve_flat, f'./data/hilbert/hilbert_curve_rank_{order}.pth')
        print(f"张量已保存为 hilbert_curve_rank_{order}.pth")
    if order+1 in orders:
        s *= 2
        curve_flat = torch.zeros(2 * s, 2 * s, device=device)
        curve_flat[:s, :s] = curve.clone().detach()
        curve_flat[:s, s:] = curve.clone().detach().permute(1, 0).flip(0) + s * s
        curve_flat[s:, s:] = curve.clone().detach().flip(0, 1) + 2 * s * s
        curve_flat[s:, :s] = curve.clone().detach().permute(1, 0).flip(1) + 3 * s * s
        curve_flat = curve_flat.view(-1)
        torch.save(curve_flat, f'./data/hilbert/hilbert_curve_es_rank_{order+1}.pth')
        print(f"张量已保存为 hilbert_curve_es_rank_{order+1}.pth")
    return curve


# 示例：生成3阶Hilbert曲线的张量并保存为pth文件
if __name__ == "__main__":
    orders = [2, 3]  # 阶数
    hilbert_curve_to_tensor(orders)

    