import torch
import numpy as np

def rays_dir_to_pixel(rays_d, height, width):
    PI = float(np.pi)
    x, y, z = rays_d[..., 0], rays_d[..., 1], rays_d[..., 2]
    theta = torch.asin(y) + (PI/2)
    phi = torch.atan2(x, z) + PI    
    pixel_y = theta * height / PI - 0.5
    pixel_x = phi * width / (2 * PI) - 0.5
    return pixel_x, pixel_y


def pixel_to_rays_dir(pixel_x, pixel_y, height, width):
    PI = np.pi
    theta = (pixel_y + 0.5) * PI / height - (PI/2)
    phi = (2 * PI * (pixel_x + 0.5) / width) - PI
    y = torch.sin(theta)
    z = torch.cos(theta) * torch.cos(phi)
    x = torch.cos(theta) * torch.sin(phi)

    rays_d = torch.stack([x, y, z], dim=-1)
    return rays_d 

def cam2world_rays(cam_rays, c2w):
    """
    cam_rays : [n_batch, 3]
    c2w : [n_batch, 3, 3]
    world_rays : [n_batch, 3]
    """
    world_rays = torch.bmm(c2w, cam_rays[:, :, None])
    return world_rays[:, :, 0]

def zaxis_front2top(in_tensor):
    """
    in_tensor(position/direction): [N, 3] tensor
    """
    rot_x = torch.tensor([
        [1, 0, 0],  # theta = -90
        [0, 0, 1],  # cos(theta), -sin(theta)
        [0, -1, 0]  # sin(theta), cos(theta)
    ], dtype=torch.float32).cuda()
    in_tensor = in_tensor[..., None].to(torch.float32)
    return torch.matmul(rot_x, in_tensor)[..., 0].to(in_tensor.dtype)