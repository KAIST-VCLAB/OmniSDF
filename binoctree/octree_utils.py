from tqdm import tqdm
import torch
from binoctree.tree import Octree

def count_hits(node):
    if node.isleaf:
        return node.hit
    else:
        hits = 0
        for ch in node.child:
            hits += count_hits(ch)
        return hits

def list_child_index(node):
    # Call only when root is not leaf
    if node.isleaf:
        return [node.idx]
    else:
        l = []
        for ch in node.child:
            l += list_child_index(ch)
        return l

def postprocess_octree(octree, conf):
    print("Post-process")
    octree.set_active_height(conf["constructor.active_height"])
    nullified = []
    for vox in octree.SPHOXEL_LIST:
        if vox.active:
            hits = count_hits(vox)
            if hits < conf["minimum_hit"]:
                nullified += list_child_index(vox)
                vox.active = False
                vox.child=[]
                vox.isleaf=True
    
    # adds
    for idx in nullified:
        octree.SPHOXEL_LIST[idx] = None



def generate_octree(octree_kwargs,
                    dataset,
                    fr_interval=1,
                    resolution_level=1,
                    debug_path=None):
    
    octree = Octree(near=dataset.near,
                    far=dataset.far,
                    **octree_kwargs)
    pts = []
    cams = []
    for img_idx in range(0, dataset.n_images, fr_interval):
        rays_o, rays_v, _, depth, mask = dataset.gen_discrete_rays_at(img_idx, resolution_level)
        mask = torch.logical_and(mask, depth < dataset.far)
        ray_mask = mask[..., None].repeat(1, 1, 3)
        pt = rays_o[ray_mask].reshape(-1, 3) + rays_v[ray_mask].reshape(-1, 3) * depth[mask].reshape(-1, 1)
        pts.append(pt.reshape(-1, 3))
        cams.append(rays_o[0, 0, :])

    pts = torch.cat(pts, dim=0).detach().cpu().numpy()
    cams = torch.stack(cams, dim=0).detach().cpu().numpy()

    with tqdm(pts) as pts_lis:
        for pt in pts_lis:
            octree.root.subdivide(pt[0], pt[1], pt[2])
    
    return octree