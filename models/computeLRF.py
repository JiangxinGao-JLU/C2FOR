import torch
import math

torch.pi = math.pi

def compute_neighborhood(points, k=60):
    """
    计算每个点的邻域点集（在 GPU 上）。

    Parameters:
    - points: 点云数据，形状为 [bs, num_corr, 3]
    - k: 邻域点的数量

    Returns:
    - neighbors: 邻域点集，形状为 [bs, num_corr, k, 3]
    """
    bs, num_corr, _ = points.shape
    neighbors = torch.zeros((bs, num_corr, k, 3), device=points.device)

    # 计算每个点对之间的欧氏距离
    for i in range(bs):
        dists = torch.cdist(points[i].unsqueeze(0), points[i].unsqueeze(0), p=2).squeeze(0)  # [num_corr, num_corr]

        # 对距离排序，获取最近的 k 个邻居的索引
        _, knn_indices = torch.topk(dists, k=k, dim=-1, largest=False)

        # 根据索引获取邻居点
        neighbors[i] = points[i][knn_indices]

    return neighbors


def compute_normal_batch(points, neighbors):
    """
    批量计算点的法向量 (LRF 的 Z 轴)

    Parameters:
    - points: 目标点，形状为 [bs, num_corr, 3]
    - neighbors: 邻域点集，形状为 [bs, num_corr, k, 3]

    Returns:
    - normals: 法向量，形状为 [bs, num_corr, 3]
    """

    # 计算每个点的邻域点的中心
    centroid = torch.mean(neighbors, dim=2, keepdim=True)

    # 计算偏移
    diff = neighbors - centroid

    # 计算协方差矩阵
    covariance = torch.matmul(diff.permute(0, 1, 3, 2), diff)  # 形状 [bs, num_corr, 3, 3]

    # 计算协方差矩阵的特征值和特征向量
    eigvals, eigvecs = torch.linalg.eigh(covariance)

    min_eigvals_indices = torch.argmin(eigvals, dim=-1)# [bs, num_corr]
    # b n 3 3 的第一个3表示特征向量的维度，第二个3表示特征向量的个数，因此需要扩展到3维选择第一个3的所有维度，而只需要选择一个特征向量
    min_eigvals_indices = min_eigvals_indices.unsqueeze(-1).expand(-1, -1, 3)

    # 选择最小特征值对应的特征向量，先将最小特征值的索引扩到b n 3 1，然后根据索引选择特征向量,最后挤压掉最后一个维度，保证法向量是b n 3
    normals = torch.gather(eigvecs, dim=-1, index=min_eigvals_indices.unsqueeze(-1)).squeeze(-1)
    return normals


def compute_LRF_batch(src_keypts, tgt_keypts, neighbors_src, neighbors_tgt, src_normals, tgt_normals):
    """
    批量计算源点云和目标点云的局部参考框架（LRF）

    Parameters:
    - src_keypts: 源点云的关键点，形状为 [bs, num_corr, 3]
    - tgt_keypts: 目标点云的关键点，形状为 [bs, num_corr, 3]
    - neighbors_src: 源点云的邻域点集，形状为 [bs, num_corr, k, 3]
    - neighbors_tgt: 目标点云的邻域点集，形状为 [bs, num_corr, k, 3]

    Returns:
    - LRF_src: 源点云的 LRF，形状为 [bs, num_corr, 3, 3]
    - LRF_tgt: 目标点云的 LRF，形状为 [bs, num_corr, 3, 3]
    """
    z_axis_src = src_normals
    z_axis_tgt = tgt_normals

    # 选择一个方向向量（与法向量正交）
    vec_src = neighbors_src[:, :, 20, :] - src_keypts  # [bs, num_corr, 3]
    vec_src = vec_src / (torch.norm(vec_src, dim=-1, keepdim=True) + 1e-8)  # 添加一个小偏移量避免除以零
    vec_tgt = neighbors_tgt[:, :, 20, :] - tgt_keypts  # [bs, num_corr, 3]
    vec_tgt = vec_tgt / (torch.norm(vec_tgt, dim=-1, keepdim=True) + 1e-8)  # 添加一个小偏移量避免除以零

    x_axis_src = torch.cross(z_axis_src, vec_src, dim=-1)
    x_axis_src = x_axis_src / (torch.norm(x_axis_src, dim=-1, keepdim=True) + 1e-8)  # 添加一个小偏移量避免除以零

    x_axis_tgt = torch.cross(z_axis_tgt, vec_tgt, dim=-1)
    x_axis_tgt = x_axis_tgt / (torch.norm(x_axis_tgt, dim=-1, keepdim=True) + 1e-8)  # 添加一个小偏移量避免除以零

    # 计算第三个轴
    y_axis_src = torch.cross(z_axis_src, x_axis_src, dim=-1)
    y_axis_tgt = torch.cross(z_axis_tgt, x_axis_tgt, dim=-1)

    # 组合成LRF矩阵
    LRF_src = torch.stack((x_axis_src, y_axis_src, z_axis_src), dim=-1)
    LRF_tgt = torch.stack((x_axis_tgt, y_axis_tgt, z_axis_tgt), dim=-1)

    return LRF_src, LRF_tgt


def compute_LRF_affinity(Lpi, Lpj):
    """
    计算 LRF affinity term (公式 8 和 9)

    Parameters:
    - Lpi: 源点的 LRF，形状为 [bs, num_corr, 3, 3]
    - Lpj: 目标点的 LRF，形状为 [bs, num_corr, 3, 3]

    Returns:
    - L(ci, cj): LRF affinity term，形状为 [bs, num_corr]
    """
    device = Lpi.device
    index = device.index

    gpu_num1 = 3
    gpu_num2 = 2

    # 计算 ϵ_lrf(p_i, p_j) = acos((trace(Lpi @ Lpj^T) - 1) / 2)
    Lpi = Lpi.cuda(0)
    Lpj = Lpj.cuda(0)
    Lpi_expanded = Lpi.unsqueeze(1)

    # Lpj_expanded 的形状为 (b, 1, n, 3, 3)
    Lpj_expanded = Lpj.unsqueeze(1)
    trace_value_src = torch.matmul(Lpi.unsqueeze(2), Lpi_expanded.permute(0, 1, 2, 4, 3)).diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    trace_value_src = torch.clamp((trace_value_src - 1) / 2, min=-1, max=1)
    epsilon_lrf_src = torch.acos(trace_value_src) * (180.0 / torch.pi)
    trace_value_tgt = torch.matmul(Lpj.unsqueeze(2), Lpj_expanded.permute(0, 1, 2, 4, 3)).diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    trace_value_tgt = torch.clamp((trace_value_tgt - 1) / 2, min=-1, max=1)
    epsilon_lrf_tgt = torch.acos(trace_value_tgt) * (180.0 / torch.pi)
    # 计算 |ϵ_lrf(p_i, p_j) - ϵ_lrf(p'_i, p'_j)|
    epsilon_lrf_src = epsilon_lrf_src.to(device)
    epsilon_lrf_tgt = epsilon_lrf_tgt.to(device)
    L_affinity = torch.abs(epsilon_lrf_src - epsilon_lrf_tgt)

    return L_affinity


def compute_angles_between_normals(normals):
    """
    计算点云中每个法向量与其他所有法向量之间的夹角，并返回一个 (b, n, n) 矩阵。

    参数:
    - normals: 法线张量，形状为 (b, n, 3)，其中 b 是批次大小，n 是点数，3 是维度（x, y, z）。

    返回:
    - angles: 形状为 (b, n, n) 的矩阵，包含所有法向量对之间的夹角（弧度）。
    """
    b, n, _ = normals.shape

   #计算内积，即矩阵乘法
    dot_products = torch.matmul(normals,normals.permute(0, 2, 1))
    dot_products = torch.clamp(dot_products, min=-1, max=1)
    # #计算模长
    # norms = torch.norm(normals, dim=2, keepdim=True).expand(b, n, n)  # 形状 (b, n, n)
    # norm_products = norms * norms.transpose(1, 2)  # 形状 (b, n, n)

    # # 计算余弦值
    # cosines = dot_products / (norm_products + 1e-6)
    # 计算反余弦值得到角度（弧度）
    angles = torch.acos(dot_products)

    return angles
