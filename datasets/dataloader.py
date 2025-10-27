import numpy as np
import torch
import random


def collate_fn(list_data):
    min_num = 1e10
    # clip the pair having more correspondence during training.
    for ind, (src_pcd, tgt_pcd, corr_pos, src_keypts, tgt_keypts, src_normals, tgt_normals, gt_trans, gt_labels) in enumerate(list_data):
        if len(gt_labels) < min_num:
            min_num = min(min_num, len(gt_labels))
    batched_src_pcd = []
    batched_tgt_pcd = []
    batched_corr_pos = []
    batched_src_keypts = []
    batched_tgt_keypts = []
    batched_src_normals = []
    batched_tgt_normals = []
    batched_gt_trans = []
    batched_gt_labels = []
    for ind, (src_pcd, tgt_pcd, corr_pos, src_keypts, tgt_keypts, src_normals, tgt_normals, gt_trans, gt_labels) in enumerate(list_data):
        sel_ind = np.random.choice(len(gt_labels), min_num, replace=False)
        batched_src_pcd.append(src_pcd[None,:,:])
        batched_tgt_pcd.append(tgt_pcd[None,:,:])
        batched_corr_pos.append(corr_pos[sel_ind, :][None,:,:])
        batched_src_keypts.append(src_keypts[sel_ind, :][None,:,:])
        batched_tgt_keypts.append(tgt_keypts[sel_ind, :][None,:,:])
        batched_src_normals.append(src_normals[sel_ind, :][None,:,:])
        batched_tgt_normals.append(tgt_normals[sel_ind, :][None,:,:])
        batched_gt_trans.append(gt_trans[None,:,:])
        batched_gt_labels.append(gt_labels[sel_ind][None, :])
    batched_src_pcd = torch.from_numpy(np.concatenate(batched_src_pcd, axis=0))
    batched_tgt_pcd = torch.from_numpy(np.concatenate(batched_tgt_pcd, axis=0))
    batched_corr_pos = torch.from_numpy(np.concatenate(batched_corr_pos, axis=0))
    batched_src_keypts = torch.from_numpy(np.concatenate(batched_src_keypts, axis=0))
    batched_tgt_keypts = torch.from_numpy(np.concatenate(batched_tgt_keypts, axis=0))
    batched_src_normals = torch.from_numpy(np.concatenate(batched_src_normals, axis=0))
    batched_tgt_normals = torch.from_numpy(np.concatenate(batched_tgt_normals, axis=0))
    batched_gt_trans = torch.from_numpy(np.concatenate(batched_gt_trans, axis=0))
    batched_gt_labels = torch.from_numpy(np.concatenate(batched_gt_labels, axis=0))
    return batched_src_pcd, batched_tgt_pcd, batched_corr_pos, batched_src_keypts, batched_tgt_keypts, batched_src_normals, batched_tgt_normals, batched_gt_trans, batched_gt_labels


def get_dataloader(dataset, batch_size, shuffle=True, num_workers=4, fix_seed=True):
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=num_workers, 
    )