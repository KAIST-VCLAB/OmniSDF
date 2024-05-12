import os, sys
import torch
import numpy as np
from utils.metric import Timer
import octree_clib



timer = Timer()


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    device = weights.device

    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])
    u = u.to(device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class Sampler:
    def __init__(self,
                 octree,
                 n_sphere=0,
                 n_sphere_dist='linear',
                 n_coarse=0,
                 n_coarse_dist='linear',
                 n_coarse_boundary=0.0,
                 n_fine=0,
                 n_fine_dist='linear',
                 n_fine_iter=0,
                 n_importance=0,
                 n_outside=0,
                 up_sample_steps=4,
                 perturb=1.0):
        
        self.octree_near = octree.near
        self.octree_far = octree.far

        self.n_sphere = n_sphere
        self.n_sphere_dist = n_sphere_dist

        self.n_coarse = n_coarse
        self.n_coarse_dist = n_coarse_dist
        self.n_coarse_boundary = n_coarse_boundary

        self.n_fine = n_fine
        self.n_fine_dist = n_fine_dist
        self.n_fine_iter = n_fine_iter

        self.n_importance = n_importance
        self.up_sample_steps = up_sample_steps

        self.n_outside = n_outside
        
        self.perturb = perturb
        
        print(f"Sampler ver2. Created :: Sphere {self.n_sphere:>2d} + Coarse {self.n_coarse:>2d} + Fine {self.n_fine:>2d} + Importance {self.n_importance:>2d}")
        
    
    @torch.no_grad()
    def n_core_samples(self, train_iter=-1):
        n_samples = self.n_sphere + self.n_coarse + self.n_fine + self.n_importance
        return n_samples
    
    @torch.no_grad()
    def n_samples(self, train_iter=-1):
        n_samples = self.n_sphere + self.n_coarse + self.n_fine + self.n_importance + self.n_outside
        return n_samples

    @torch.no_grad()
    def z_grid(self, n_samples, distribution):
        if distribution == 'linear':
            z_vals = torch.linspace(0.0, 1.0, n_samples).cuda()
        elif distribution == 'lindisp':
            z_vals = 1.0/torch.linspace(n_samples, 1.0, n_samples).cuda()
        else:
            sys.exit("invalid argument")
        return z_vals

    @torch.no_grad()
    def sample_by_sphere(self, rays_o, rays_d, near, far, n_samples, dist):
        near = near[..., None]
        far = far[..., None]

        if self.perturb > 0:
            z_vals = near + (far - near) * self.z_grid(n_samples + 1, dist)[None, :]
            z_dist = z_vals[:, 1:] - z_vals[:, :-1]
            z_mid =  (z_vals[:, 1:] + z_vals[:, :-1]) / 2
            t_rand = torch.rand(len(z_dist), 1).cuda() - 0.5
            z_vals = z_mid + z_dist * t_rand
        else:
            z_vals = near + (far - near) * self.z_grid(n_samples, dist)[None, :]
        
        return z_vals

    
    @torch.no_grad()
    def sample_by_voxel(self, rays_o, rays_d, near, far, tree_vertex, tree_childs, sample_mask, n_samples):
        MAX_DEPTH = 10000.0
        
        n_coarse = n_samples + 1 if self.perturb > 0 else n_samples
        n_coarse = n_samples

        out = octree_clib.tree_cuboid_intersection(rays_o,
                                                   rays_d,
                                                   tree_vertex,
                                                   tree_childs,
                                                   sample_mask,
                                                   n_coarse)
        
        intersect_idx, min_depths, max_depths = out
        
        hits = intersect_idx.ne(-1).any(-1)
        invalid = intersect_idx.eq(-1)
        intersect_idx.masked_fill_(invalid, 0)
        min_depths.masked_fill_(invalid, MAX_DEPTH)
        max_depths.masked_fill_(invalid, MAX_DEPTH)
        
        min_depths, sorted_idx = min_depths.sort(dim=-1)
        max_depths = max_depths.gather(-1, sorted_idx)
        intersect_idx = intersect_idx.gather(-1, sorted_idx)
        sample_dist = torch.sum(max_depths - min_depths, -1) / n_coarse
        
        voxel_sample, sample_dist_out = octree_clib.uniform_sampling(min_depths,
                                                                 max_depths,
                                                                 sample_dist,
                                                                 near,
                                                                 far,
                                                                 hits.to(torch.float32),
                                                                 n_coarse,
                                                                 self.n_coarse_boundary)
        
        if self.perturb > 0:
            rand = torch.rand(rays_o.size(0), 1).cuda()
            voxel_sample = voxel_sample[..., :-1] + rand * sample_dist[:, None]
        
        
        return voxel_sample
        

    @torch.no_grad()
    def sample(self, rays_o, rays_d,
               sdf_network=None,
               tree_vertex=None,
               tree_childs=None,
               coarse_sample_mask=None,
               fine_sample_mask=None,
               train_iter=-1, DEBUG=False):
        
        if DEBUG: debug_z = []
        
        near = octree_clib.get_sphere_intersection(rays_o, rays_d, self.octree_near)
        far = octree_clib.get_sphere_intersection(rays_o, rays_d, self.octree_far)
        
        sphere_samples = []
        if self.n_sphere > 0:
            if DEBUG: timer.tick()
            z_sphere = self.sample_by_sphere(rays_o,
                                             rays_d,
                                             near,
                                             far,
                                             self.n_sphere,
                                             self.n_sphere_dist).detach()
            sphere_samples.append(z_sphere)

            if DEBUG: timer.tick("Sample sphere")
            if DEBUG: debug_z.append(z_sphere)
        
        binoctree_samples = []
        if self.n_coarse > 0:
            if DEBUG: timer.tick()
            n_coarse = coarse_sample_mask.size(0)
            z_sphoxel = self.sample_by_voxel(rays_o,
                                             rays_d,
                                             near,
                                             far,
                                             tree_vertex[:n_coarse],
                                             tree_childs[:n_coarse],
                                             coarse_sample_mask,
                                             self.n_coarse).detach()
            binoctree_samples.append(z_sphoxel)

            if DEBUG: timer.tick("Sample coarse voxel")
            if DEBUG: debug_z.append(z_sphoxel)
        
        if self.n_fine > 0:
            if DEBUG: timer.tick()
            near = z_sphoxel[:, 0]
            far = z_sphoxel[:, -1]
            z_surface = self.sample_by_voxel(rays_o,
                                             rays_d,
                                             near,
                                             far,
                                             tree_vertex,
                                             tree_childs,
                                             fine_sample_mask,
                                             self.n_fine).detach()
            binoctree_samples.append(z_surface)
            
            if DEBUG: timer.tick("Sample fine voxel")
            if DEBUG: debug_z.append(z_surface)
        

        if self.n_importance > 0:
            if self.n_coarse >0 or self.n_fine > 0:
                z_vals = torch.cat(binoctree_samples, dim=-1)
                z_vals, _ = torch.sort(z_vals, dim=-1)
            else:
                z_vals = z_sphere
            
            batch_size = len(rays_o)
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
            n_samples = pts.size(1)
            
            sdf = sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_samples)
            
            for i in range(self.up_sample_steps):
                if DEBUG: timer.tick()
                new_z_vals = self.up_sample(rays_o,
                                            rays_d,
                                            z_vals,
                                            sdf,
                                            self.n_importance // self.up_sample_steps,
                                            64 * 2**i)
                if DEBUG: timer.tick("Up sample")
                if DEBUG: debug_z.append(new_z_vals)

                z_vals, sdf = self.cat_z_vals(rays_o,
                                            rays_d,
                                            z_vals,
                                            new_z_vals,
                                            sdf,
                                            sdf_network,
                                            last=(i + 1 == self.up_sample_steps))
            
        if self.n_coarse >0 or self.n_fine > 0:
            z_vals = torch.cat(sphere_samples + [z_vals], dim=-1)
            z_vals, _ = torch.sort(z_vals, dim=-1)
            
        if DEBUG: return z_vals, sphere_samples + binoctree_samples

        return z_vals
    
    @torch.no_grad()
    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < self.octree_far) | (radius[:, 1:] < self.octree_far)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]).to(cos_val.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]).to(alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples
    
    @torch.no_grad()
    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, sdf_network, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf
    
    @torch.no_grad()
    def sample_outside(self, batch_size, far=1.0):
        n_outside = self.n_outside
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (n_outside + 1.0), n_outside)
            if self.perturb > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_sphere
            return z_vals_outside
        else:
            return None