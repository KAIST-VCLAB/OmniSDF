import os
from utils.pcd import PointCloud
import pickle
import numpy as np
from contextlib import redirect_stdout

def save_octree_vertices(octree, save_path, save_name, fn=lambda x: True):
    points = []
    for vox in octree.SPHOXEL_LIST:
        if vox is not None and fn(vox):
            points.append(vox.get_vertices_xyz())

    pcd = PointCloud()
    pcd.add_pts(np.concatenate(points, axis=0))
    pcd.gen_ply(save_path, save_name)


def save_octree_pickle(octree, save_path, octree_name):
    octree_path = os.path.join(save_path, f"{octree_name}.pickle")
    with open(octree_path, "wb") as handle:
        pickle.dump(octree, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_octree_ckpts(octree, save_path, octree_name):
    octree.clear_leaf()
    save_octree_pickle(octree, save_path, octree_name)
    save_octree_vertices(
        octree,
        save_path,
        f"{octree_name}_vertices",
        fn=lambda vox: vox.isleaf)


def print_tree(node, last=True, header=''):
    elbow = "└──"
    pipe = "│  "
    tee = "├──"
    blank = "   "
    print(header + (elbow if last else tee) + f"[{node.level}] {node.idx}    R {node.min_r:.4f} {node.max_r:.4f}    Theta {node.min_theta:.4f} {node.max_theta:.4f}    PHI {node.min_phi:.4f} {node.max_phi:.4f}")
    if not node.isleaf:
        N_children = len(node.child)
        for i, c in enumerate(node.child):
            print_tree(c, header=header + (blank if last else pipe), last=i == N_children - 1)


def print_tree_structure(octree, save_path):
    with open(os.path.join(save_path, "tree.txt"), 'w') as f:
        with redirect_stdout(f):
            print_tree(octree.root)


