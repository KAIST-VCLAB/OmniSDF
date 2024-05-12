import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os, sys
from glob import glob
# from icecream import ic
# from scipy.spatial.transform import Rotation as Rot
# from scipy.spatial.transform import Slerp
import csv, math
from models.dataloader.dataset_utils import *

class Egocentric360:
    def __init__(self, conf):
        super(Egocentric360, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.data_dir = conf.get_string('data_dir')
        self.camera_outside_sphere = True

        self.fr_start = conf.get_int('fr_start')
        self.fr_end = conf.get_int('fr_end')
        self.fr_interval = conf.get_int('fr_interval')
        self.fr_scale = conf.get_float('fr_scale')
        self.world_scale = conf.get_float('world_scale')
        self.far_sphere_bound = conf.get_float('far_sphere_bound')

        try:
            self.world_shift = conf.get_float('world_shift')
        except:
            self.world_shift = 0
        
        # images
        frames = []
        cap = cv.VideoCapture(os.path.join(self.data_dir, "video.mp4"))
        for frame_id in range(self.fr_start, self.fr_end, self.fr_interval):
            if not cap.isOpened():
                print("Error opening video stream or file")
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            if frame is not None:
                frames.append(
                    cv.resize(
                        frame, dsize=(0, 0),
                        fx=self.fr_scale, fy=self.fr_scale, interpolation=cv.INTER_AREA
                        )
                    )
        cap.release()
        self.n_images = len(frames)
        self.images_np = np.stack(frames) / 256.0
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cuda()  # [n_images, H, W, 3]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W
        
        # masks
        mask_np = cv.imread(os.path.join(self.data_dir, "mask_img.png"))[..., 0] / 256.0
        self.mask = torch.from_numpy(cv.resize(mask_np, dsize=(self.W, self.H)) > 0.5).cuda()   # Valid pixel
        
        # depths
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

        self.depths_lis = [os.path.join(self.data_dir, "idepth", f"{frame_id}.exr") for frame_id in range(self.fr_start, self.fr_end, self.fr_interval)]
        print("Image resolution", self.H, self.W)

        self.depths_np = np.stack([cv.resize(cv.imread(im_name, cv.IMREAD_UNCHANGED), dsize=(self.W, self.H)) for im_name in self.depths_lis])
        self.idepths = torch.from_numpy(self.depths_np.astype(np.float32)).cuda()
        
        invalid_depth_msk = torch.logical_or((self.idepths < 1e-4), ~self.mask)
        depths = 1.0 / self.idepths
        self.idepths[invalid_depth_msk] = 0.0
        depths[invalid_depth_msk] = 0.0
        
        # depths = torch.nan_to_num(depths, 0)

        print("depth min, max", torch.max(depths), torch.min(depths))

        self.depths = depths
        self.depths_mask = ~invalid_depth_msk

        traj = []
        with open(os.path.join(self.data_dir, "traj.csv")) as f:
            csv_reader = csv.reader(f, delimiter=" ")
            for row in csv_reader:
                mat = np.array(row[1:], dtype=np.float32).reshape((4, 4))   # c2w [4, 4]
                mat = np.linalg.inv(mat)                                    # w2c
                traj.append(mat)

        traj_full = np.stack(traj, axis=0)[:, :3, -1]
        self.cam_center = np.mean(traj_full, axis=0)
        traj_full = traj_full - self.cam_center
        self.near = np.max(np.sqrt(traj_full[..., 0]**2 + traj_full[..., 1]**2 + traj_full[..., 2]**2))
        self.far = self.near / math.tan(math.pi / self.W) if self.far_sphere_bound < 0 else self.far_sphere_bound

        # Select camera
        self.traj = np.stack([traj[num] for num in range(self.fr_start, self.fr_end, self.fr_interval)], axis=0)   # [N, 4, 4]
        self.cam_pos = self.traj[:, :3, -1] - self.cam_center[None, ...]    # Centeralize [N, 3]
        self.cam_pos[:, :2] += self.world_shift
        self.cam_rot = self.traj[:, :3, :3]
        self.pose_all = torch.tensor(self.cam_rot).cuda()

        # Set bounding sphere
        
        print("Bounding sphere", self.near, self.far)
        self.sphere_scale = self.far
        self.cam_pos /=  self.far
        self.depths /= self.far
        self.near /= self.far
        self.far /= self.far
        print("Bounding sphere after normalization", self.near, self.far)
        
        try:
            self.object_bbox_min = np.array(conf['obj_bbox_min'])
            self.object_bbox_max = np.array(conf['obj_bbox_max'])
        except:
            self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
            self.object_bbox_max = np.array([1.01, 1.01, 1.01])

        print('Load data: End')


    def gen_rays_at(self, img_idx, resolution_level=1, debug=False):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        pixels_x = pixels_x.transpose(0, 1)
        pixels_y = pixels_y.transpose(0, 1)
        
        # w = self.W//l
        # h = self.H//l
        h = pixels_x.size(0)
        w = pixels_x.size(1)

        rays_o = torch.tensor(np.tile(self.cam_pos[img_idx:img_idx+1, ...], (w*h, 1))).cuda()
        
        rays_rot = torch.tensor(np.tile(self.cam_rot[img_idx:img_idx+1, ...], (w*h, 1, 1))).contiguous().cuda()  # [N, 3, 3]
        rays_v = pixel_to_rays_dir(pixels_x, pixels_y, self.H, self.W).reshape(-1, 3).contiguous().cuda()  # [N=H*W, 3]
        # rays_v = pixel_to_rays_dir(pixels_x, pixels_y, h, w).reshape(-1, 3).contiguous().cuda()  # [N=H*W, 3]
        rays_v = cam2world_rays(rays_v, rays_rot)
        
        rays_o = zaxis_front2top(rays_o)
        rays_v = zaxis_front2top(rays_v)
        rays_o = rays_o.reshape(h, w, 3)
        rays_v = rays_v.reshape(h, w, 3)

        rays_v = torch.nn.functional.normalize(rays_v, dim=-1)
        
        return rays_o, rays_v
        # depth = self.depths[img_idx]    
        # return rays_o, rays_v, depth
    
    def gen_discrete_rays_at(self, img_idx, resolution_level=1):
        # l = resolution_level
        tx = torch.arange(0, self.W - 1, resolution_level) #l)
        ty = torch.arange(0, self.H - 1, resolution_level) #l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty) # [W, H]
        pixels_x = pixels_x.transpose(0, 1)         # [H, W]
        pixels_y = pixels_y.transpose(0, 1)

        #mask = self.depths_mask[img_idx][(pixels_y, pixels_x)]      # [H, W]
        mask = torch.logical_and(
            self.depths_mask[img_idx][(pixels_y, pixels_x)],
            self.mask[(pixels_y, pixels_x)]
        )
        depth = self.depths[img_idx][(pixels_y, pixels_x)]
        
        h = pixels_x.size(0)
        w = pixels_x.size(1)

        rays_o = torch.tensor(np.tile(self.cam_pos[img_idx:img_idx+1, ...], (w*h, 1))).cuda()
        
        rays_rot = torch.tensor(np.tile(self.cam_rot[img_idx:img_idx+1, ...], (w*h, 1, 1))).contiguous().cuda()  # [N, 3, 3]
        # rays_v = pixel_to_rays_dir(pixels_x, pixels_y, h, w).reshape(-1, 3).contiguous().cuda()  # [N=H*W, 3]
        # rays_v = pixel_to_rays_dir(pixels_x, pixels_y, self.H, self.W).reshape(-1, 3).contiguous().cuda()  # [N=H*W, 3]
        rays_v = pixel_to_rays_dir(pixels_x, pixels_y, self.H, self.W).reshape(-1, 3).contiguous().cuda()  # [N=H*W, 3]
        rays_v = cam2world_rays(rays_v, rays_rot)
        
        rays_o = zaxis_front2top(rays_o)
        rays_v = zaxis_front2top(rays_v)
        rays_o = rays_o.reshape(h, w, 3)
        rays_v = rays_v.reshape(h, w, 3)

        rays_v = torch.nn.functional.normalize(rays_v, dim=-1)

        color = self.images[img_idx][(pixels_y, pixels_x)]
        
        return rays_o, rays_v, color, depth, mask


    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """

        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.mask[(pixels_y, pixels_x)]      # batch_size, 3
        depth = self.depths[img_idx][(pixels_y, pixels_x)]

        rays_o = torch.tensor(np.tile(self.cam_pos[img_idx:img_idx+1, ...], (batch_size, 1))).cuda()
        
        rays_rot = torch.tensor(np.tile(self.cam_rot[img_idx:img_idx+1, ...], (batch_size, 1, 1))).contiguous().cuda()  # [N, 3, 3]
        rays_v = pixel_to_rays_dir(pixels_x, pixels_y, self.H, self.W).reshape(-1, 3).contiguous().cuda()  # [N=H*W, 3]
        rays_v = cam2world_rays(rays_v, rays_rot)

        rays_o = zaxis_front2top(rays_o)
        rays_v = zaxis_front2top(rays_v)
        rays_v = torch.nn.functional.normalize(rays_v, dim=-1)

        return torch.cat([rays_o, rays_v, color.cuda(), mask[..., None], depth[..., None]], dim=-1)    # batch_size, 10 + 1


    def image_at(self, idx, resolution_level):
        return (cv.resize(self.images_np[idx]*256, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
        # return (self.images_np[idx] * 256.0).clip(0, 255)
