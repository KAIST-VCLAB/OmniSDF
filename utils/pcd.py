import open3d as o3d
import numpy as np
import torch
from os.path import join

class PointCloud:
    def __init__(self, name=None):
        self.name=name
        self.pcd = o3d.geometry.PointCloud()
        self.pts = []
        self.colors = []
    
    def add_pt(self, pt):
        self.pts.append(pt[None, ...])

    def add_pts(self, pt):
        self.pts.append(pt)

    def clear_and_add_pt(self, pt):
        self.pts = [pt[None, ...]]

    def clear_and_add_pts(self, pt):
        self.pts = [pt]

    def add_colored_pt(self, pt, color):
        self.pts.append(pt[None, ...])
        self.colors.append(color[None, ...])

    def add_colored_pts(self, pt, color):
        self.pts.append(pt)
        self.colors.append(color)

    def gen_ply(self, fpath=None, fname=None):
        if len(self.pts) >= 1:
            if len(self.pts) > 1:
                pts = np.concatenate(self.pts, axis=0)
            else:
                pts = self.pts[0]

            self.pcd.points = o3d.utility.Vector3dVector(pts)    
        else:
            return
        
        if len(self.colors) >= 1:
            if len(self.colors) > 1:
                colors = np.concatenate(self.colors, axis=0)
            else:
                colors = self.colors[0]
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            print(self.pcd.has_colors())
        
        if fpath is not None:
            ply_name = f"{fname}.ply" if fname is not None else f"{self.name}.ply"
            o3d.io.write_point_cloud(join(fpath, ply_name), self.pcd, write_ascii=True)


# Point cloud utils for raw dataloader
def dataset_to_pointcloud(dataset, out_path): # For debug purpose
    pcd = PointCloud()

    for img_idx in range(0, dataset.n_images):
        rays_o, rays_v, color, depth, mask = dataset.gen_discrete_rays_at(img_idx, 10)
        mask = torch.logical_and(mask, depth < dataset.far)
        mask3ch = mask[..., None].repeat(1, 1, 3)
        pt = rays_o[mask3ch].reshape(-1, 3) + rays_v[mask3ch].reshape(-1, 3) * depth[mask].reshape(-1, 1)
        pt = pt.reshape(-1, 3).detach().cpu().numpy()

        color = color[mask3ch]
        
        # Scene point cloud in RGB color
        pcd.add_colored_pts(pt, color.reshape(-1, 3).detach().cpu().numpy())

        # Scene point cloud in blue
        # pcd.add_colored_pts(pt, np.tile(np.array([0, 0, 1])[None, :], (pt.shape[0], 1)))

        # Camera positions
        pcd.add_colored_pt(rays_o[0, 0, :].detach().cpu().numpy(), np.array([1, 0, 0]))

    pcd.gen_ply(out_path, "raw_pcd")


