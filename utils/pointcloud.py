import open3d as o3d
import torch


def make_point_cloud(pts):
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd 

def make_feature(data, dim, npts):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    feature = o3d.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.astype('d').transpose()
    return feature

def estimate_normal(pcd, radius=0.06, max_nn=30):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))


def refine(src_keypts, tgt_keypts, pred_trans):
    """
    ICP algorithm to refine the initial transformation
    Input:
        - src_keypts [1, num_corr, 3] FloatTensor
        - tgt_keypts [1, num_corr, 3] FloatTensor
        - pred_trans [1, 4, 4] FloatTensor, initial transformation
    """
    src_pcd = make_point_cloud(src_keypts.detach().cpu().numpy()[0])
    tgt_pcd = make_point_cloud(tgt_keypts.detach().cpu().numpy()[0])
    initial_trans = pred_trans[0].detach().cpu().numpy()
    # change the convension of transforamtion because open3d use left multi.
    refined_T = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, 0.10, initial_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()).transformation
    refined_T = torch.from_numpy(refined_T[None, :, :].copy()).to(pred_trans.device).float()
    return refined_T