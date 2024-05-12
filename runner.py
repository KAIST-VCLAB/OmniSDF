import os, sys, time
import pickle
import logging
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory

import numpy as np
import cv2 as cv
import trimesh

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.dataloader.dataset_omniphoto import Egocentric360
from models.fields import *
from models.renderer import OmniSDFRenderer
from utils.pcd import *
from utils.metric import *
from binoctree.octree_utils import *
from binoctree.octree_io import *

import octree_clib

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        timer = Timer()

        self.device = torch.device('cuda')
        os.environ["CUDA_LAUNCH_BLOCKING"]="1"
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.is_continue = is_continue
        self.mode = mode

        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.debug = self.conf['general.debug']
        self.iter_step = 0
        self.end_iter = self.conf.get_int('train.end_iter')

        # Load dataset
        self.dataset_class = self.conf['general.dataset_classname']
        if self.dataset_class == 'Omniphoto':
            self.dataset = Egocentric360(self.conf['dataset'])
        else:
            sys.exit()
        
        
        if self.debug: dataset_to_pointcloud(self.dataset, self.base_exp_dir)

        # Create Octree
        if self.conf.get_int('model.sampler.n_coarse') > 0 or self.conf.get_int('model.sampler.n_fine') > 0:
            self.octree_dir = os.path.join(self.base_exp_dir, self.conf.get_string("octree.logdir"))
            os.makedirs(self.octree_dir, exist_ok=True)

            latest_tree_name = None
            if is_continue:
                self.continue_iter = self.conf.get_int('general.is_continue')
                if self.continue_iter > 0:
                    # Find tree of designated iteration
                    latest_tree_name = f"{self.continue_iter:08d}_octree.pickle"
                    latest_ckpt_name = f'octree_ckpt_{self.continue_iter:08d}.pth'
                else:
                    # Find latest tree 
                    model_list_raw = os.listdir(self.octree_dir)
                    print(self.octree_dir)
                    print(model_list_raw)
                    model_list = []
                    for model_name in model_list_raw:
                        if model_name[-14:] == '_octree.pickle' and (model_name[:8]).isdigit() and int(model_name[:8]) <= self.end_iter:
                            model_list.append(model_name)
                    model_list.sort()
                    latest_tree_name = model_list[-1]
                    latest_ckpt_name = f'octree_ckpt_{latest_tree_name[:8]}.pth'
                
                # Load tree to continue
                with open(os.path.join(self.octree_dir, latest_tree_name), "rb") as handle:
                    print("Load octree: ", latest_tree_name)
                    timer.tick()
                    self.octree = pickle.load(handle)
                    timer.tick("Load octree")
                    self.load_octree_checkpoint(latest_ckpt_name)
            else:
                self.create_octree()
                print("Start octree tensor packing")
                timer.tick()
                self.pack_coarse_voxels()
                timer.tick("End octree tensor packing")
        else:
            self.tree_vertex = None
            self.tree_childs = None
            self.coarse_sample_mask = None
            self.fine_sample_mask = None
            self.octree = Octree(near=self.dataset.near,
                                 far=self.dataset.far,
                                 **self.conf["octree.constructor"])

        # Training parameters
        
        self.summary_img = self.conf.get_bool('general.summary_image')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')

        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')

        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.save_freq = self.conf.get_int('train.save_freq')
        
        self.update_octree_freq = self.conf.get_int('train.update_octree_freq')
        self.update_octree_start = self.conf.get_int('train.update_octree_start')
        self.update_octree_end = self.conf.get_int('train.update_octree_end')


        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')

        self.model_list = []
        self.writer = None

        params_to_train = []
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        

        # Renderer
        self.renderer = OmniSDFRenderer(
            self.nerf_outside,
            self.sdf_network,
            self.deviation_network,
            self.color_network,
            sampler=self.conf['model.sampler'],
            octree=self.octree)
        
        # Load checkpoint
        latest_model_name = None
        if is_continue:
            self.continue_iter = self.conf.get_int('general.is_continue')
            if self.continue_iter > 0:
                latest_model_name = f"ckpt_{self.continue_iter:08d}.pth"
            else:
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        if self.mode[:5] == 'train':
            self.file_backup()


    def create_octree(self):
        timer = Timer()
        # 1. Load octree from path
        octree_path = self.conf["octree.path"]
        if os.path.isfile(octree_path):
            timer.tick()
            with open(octree_path, "rb") as handle:
                self.octree = pickle.load(handle)
            timer.tick("Octree load time ")
            return
            
        ####################################

        # 2. Build octree
        raw_path = os.path.join(self.octree_dir, "raw_octree.pickle")
        build_true = self.conf.get_bool("octree.force_build")
        postprocess = self.conf.get_bool("octree.postprocess")
        
        # 2.1 Create initial binoctree from the start
        if build_true or not os.path.isfile(raw_path):
            self.octree = generate_octree(self.conf["octree.constructor"],
                                          self.dataset,
                                          fr_interval = self.conf.get_int("octree.fr_interval"),
                                          resolution_level=self.conf.get_int("octree.resolution_level"),
                                          debug_path=self.octree_dir)
            save_octree_ckpts(self.octree, self.octree_dir, "raw_octree")
        else:
            with open(raw_path, 'rb') as handle:
                self.octree = pickle.load(handle)

        # 2.2 Postprocess from raw binoctree
        postprocess_octree(self.octree, self.conf["octree"])
        
        # Save create octree
        octree_name = f"{self.iter_step:08d}_octree"
        print_tree_structure(self.octree, self.octree_dir)
        save_octree_ckpts(self.octree, self.octree_dir, octree_name)

        if self.debug:
            save_octree_vertices(self.octree,
                                self.octree_dir,
                                "coarse_sample_vertices",
                                fn=lambda vox: vox.active)
            n_voxels, n_vertices = self.octree.count_voxels()
            print(f"Initial octree: voxels\t{n_voxels}\tvertices\t{n_vertices}")
            print(f"Indexing:       voxels\t{self.octree.VOXEL_CNT}\tvertices\t{len(self.octree.v)}")
    
    
    def train(self):
        torch.cuda.empty_cache()

        timer = Timer()

        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):    
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            with torch.autograd.detect_anomaly(False):
                if self.dataset_class == 'Public':
                    near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
                else:
                    near, far = -1, -1

                sample_kwargs = {
                "tree_vertex": self.tree_vertex,
                "tree_childs": self.tree_childs,
                "coarse_sample_mask": self.coarse_sample_mask,
                "fine_sample_mask": self.fine_sample_mask
                }
                render_out = self.renderer.render(rays_o.contiguous(),
                                                  rays_d.contiguous(),
                                                  near=near, far=far,
                                                  background_rgb=background_rgb,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                  sample_kwargs=sample_kwargs,
                                                  iter_step=self.iter_step)                

            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            loss = color_fine_loss +\
                eikonal_loss * self.igr_weight

            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            loss = color_fine_loss +\
                eikonal_loss * self.igr_weight +\
                mask_loss * self.mask_weight
            
            self.optimizer.zero_grad()
            
            loss.backward()
            
            self.optimizer.step()

            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
            
            self.iter_step += 1

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))
            
            if self.iter_step % self.val_freq == 0:
                self.validate_image(0)

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            if self.update_octree_freq > 0 and self.iter_step % self.update_octree_freq == 0:
                if self.iter_step >= self.update_octree_start and self.iter_step < self.update_octree_end:
                    self.subdivide_octree()

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()
    
    @torch.no_grad()
    def pack_coarse_voxels(self):
        
        tree_specs, tree_childs = self.octree.pack_tree_tensor()

        self.tree_specs = tree_specs                                    # [N, 6]
        self.tree_vertex = self.octree.compute_node_vertice(tree_specs) # [N, 8, 3]
        self.tree_childs = tree_childs                                  # [N, 9]

        self.N_coarse = self.octree.VOXEL_CNT

        # Coarse voxels mask
        sample_mask = torch.as_tensor(
            [vox is not None and vox.active for vox in self.octree.SPHOXEL_LIST],
            dtype=torch.bool, device="cuda:0")
        self.coarse_sample_mask = sample_mask.to(torch.float32) # active and leaf
        self.coarse_index = torch.arange(self.octree.VOXEL_CNT,
                                         dtype=torch.int,
                                         device="cuda:0")[sample_mask]
        
        self.fine_sample_mask = self.coarse_sample_mask.clone().detach()
        self.fine_index = self.coarse_index.clone().detach()
    

    @torch.no_grad()
    def subdivide_octree(self):
        timer = Timer()
        """
        Data struectures to update : self.tree_vertex
                                     self.tree_childs
        """
        
        # Collect leaf voxels in coarse voxel
        timer.tick()
        self.octree.clear_leaf()
        for i_coarse in self.coarse_index:
            node = self.octree.SPHOXEL_LIST[i_coarse]
            self.octree.collect_leaf(node, node.level)
        print(f"Collected {len(self.octree.leaf_vec)} of leaf voxels")
        timer.tick("Subdivision : collect leaf")

        # Pack specification of collected leaf voxels
        timer.tick()
        leaf_specs = []
        leaf_index = []
        for vox in self.octree.leaf_vec:
            leaf_index.append(vox.idx)
            leaf_specs.append([vox.min_r, vox.max_r, vox.min_theta, vox.max_theta, vox.min_phi, vox.max_phi])
        leaf_specs = torch.tensor(leaf_specs, device="cuda:0", dtype=torch.float32, requires_grad=False)
        leaf_index = torch.tensor(leaf_index, device="cuda:0", dtype=torch.int, requires_grad=False)

        # Surface existence test
        vertices = self.octree.compute_node_vertice(leaf_specs)
        centers, radius = self.octree.compute_node_size(vertices)
        sdfs = self.compute_node_sdf(centers)
        surface_idx = leaf_index[torch.abs(sdfs) < radius]
        timer.tick("Subdivision: compute surface existence")
        
        # Subdivide and update tree specification
        timer.tick()
        new_specs = []
        num_child = 0
        for idx in surface_idx:
            node = self.octree.SPHOXEL_LIST[idx]
            if node.division_check():
                node.create_subtree() # new voxel created
                # Update tree_spec with new voxels
                for ch in node.child:
                    num_child += 1
                    new_specs.append(ch.get_spherical_bound())
                # Update tree_childs of subdivided voxels
                if len(node.child) == 8:
                    self.tree_childs[idx, :8] = torch.as_tensor([
                        ch.idx for ch in node.child
                    ], device="cuda:0", dtype=torch.int)
                elif len(node.child) == 8:
                    self.tree_childs[idx, :2] = torch.as_tensor([
                        ch.idx for ch in node.child
                    ], device="cuda:0", dtype=torch.int)
                self.tree_childs[idx, -1] = len(node.child)
        
        if len(new_specs) > 0:
            new_specs = torch.from_numpy(np.array(new_specs, dtype=np.float32)).cuda().detach()
            new_vertices = self.octree.compute_node_vertice(new_specs)
            
            self.tree_specs = torch.cat([self.tree_specs, new_specs], dim=0)
            self.tree_vertex = torch.cat([self.tree_vertex, new_vertices], dim=0)
            self.tree_childs = torch.cat([self.tree_childs, torch.zeros([num_child, 9], dtype=torch.int, device="cuda:0")], dim=0)
        
            assert(self.tree_childs.size(0) == self.tree_vertex.size(0))
        
        self.fine_index = surface_idx
        self.fine_sample_mask = torch.zeros(self.octree.VOXEL_CNT, dtype=torch.float32, device="cuda:0")
        self.fine_sample_mask[surface_idx.long()] = 1.0

        timer.tick("Subdivision")

        # Log
        if self.debug:
            os.makedirs(os.path.join(self.octree_dir, "log"), exist_ok=True)
            pcd = PointCloud()
            for idx in self.fine_index:
                pcd.add_pts(self.octree.SPHOXEL_LIST[idx].get_vertices_xyz())
            pcd.gen_ply(os.path.join(self.octree_dir, "log"), f"{self.iter_step:08d}_surface_voxels")


    @torch.no_grad()
    def compute_node_sdf(self, centers):
        timer = Timer()
        timer.tick()
        centers_ls = centers.split(4096)
        sdfs = []
        for centers_batch in centers_ls:
            sdf = self.sdf_network.sdf(centers_batch).detach()
            sdfs.append(sdf)
        sdfs = torch.cat(sdfs, axis=0)[:, 0]
        timer.tick("SDF computation")
        return sdfs


    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        
    def load_octree_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.octree_dir, checkpoint_name)
        if not os.path.isfile(checkpoint_path):
            print("No octree checkpoints file to load")
            return
        else:
            print(f"Load octree checkpoint : {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.tree_specs = checkpoint['tree_specs']
        self.tree_childs = checkpoint['tree_childs']
        self.N_coarse = checkpoint['n_coarse']
        self.coarse_index = checkpoint['coarse_index']
        self.fine_index = checkpoint['fine_index']

        self.tree_vertex = self.octree.compute_node_vertice(self.tree_specs)

        self.coarse_sample_mask = torch.zeros(
            self.N_coarse, dtype=torch.float32, device="cuda:0")
        self.coarse_sample_mask[self.coarse_index.long()] = 1.0
            
        self.fine_sample_mask = torch.zeros(
            self.tree_specs.size(0), dtype=torch.float32, device="cuda:0")
        self.fine_sample_mask[self.fine_index.long()] = 1.0

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>8d}.pth'.format(self.iter_step)))
        
        if self.conf.get_int('model.sampler.n_coarse') > 0 or self.conf.get_int('model.sampler.n_fine') > 0:
            octree_ckpt = {
                'tree_specs' : self.tree_specs,
                'tree_childs' : self.tree_childs,
                'n_coarse': self.N_coarse,
                'coarse_index' : self.coarse_index,
                'fine_index': self.fine_index
            }
            if self.conf.get_int('model.sampler.n_fine')==0:
                save_octree_ckpts(self.octree, self.octree_dir, "00000000_octree")
                torch.save(octree_ckpt, os.path.join(self.octree_dir, 'octree_ckpt_00000000.pth'))
            else:
                save_octree_ckpts(self.octree, self.octree_dir, f"{self.iter_step:08d}_octree")
                torch.save(octree_ckpt, os.path.join(self.octree_dir, 'octree_ckpt_{:0>8d}.pth'.format(self.iter_step)))
    

    def validate_image(self, idx=-1, resolution_level=-1, debug=False):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        print(H, W)
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size * 4)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size * 4)

        out_rgb_fine = []
        out_normal_fine = []
        
        # Feed forward
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            if self.dataset_class == 'Public':
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            else:
                near, far = -1, -1
            
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None
            sample_kwargs = {
                "tree_vertex": self.tree_vertex,
                "tree_childs": self.tree_childs,
                "coarse_sample_mask": self.coarse_sample_mask,
                "fine_sample_mask": self.fine_sample_mask
            }
            render_out = self.renderer.render(rays_o_batch.contiguous(),
                                              rays_d_batch.contiguous(),
                                              near=near, far=far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              sample_kwargs=sample_kwargs,
                                              iter_step=self.iter_step)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.sampler.n_core_samples(self.iter_step)
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)

            del render_out
        
        # Stack output
        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.array([[1, 0, 0],  # theta = -90
                            [0, 0, 1],  # cos(theta), -sin(theta)
                            [0, -1, 0]] # sin(theta), cos(theta)
                            ).astype(np.float32)
            rot = np.linalg.inv(rot)
            normal_img = np.matmul(rot[None, :, :], normal_img[:, :, None])
            normal_img = (normal_img.reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
        
        # Log image to summary writer
        if self.mode == 'train':
            self.writer.add_image("rgb_map", cv.cvtColor(img_fine[..., 0]/255.0, cv.COLOR_BGR2RGB) , self.iter_step, dataformats='HWC')
            self.writer.add_image("normal_map", cv.cvtColor(normal_img[..., 0]/255.0, cv.COLOR_BGR2RGB), self.iter_step, dataformats='HWC')

        # Save rgb and normal map in server
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        if debug:
            os.makedirs(os.path.join(self.base_exp_dir, 'validate_render'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                if debug:
                    img_path = os.path.join(
                        self.base_exp_dir,
                        'validate_render',
                        'rgb_{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx))
                else:
                    img_path = os.path.join(
                        self.base_exp_dir,
                        'validations_fine',
                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx))

                cv.imwrite(img_path,
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                if debug:
                    img_path = os.path.join(
                        self.base_exp_dir, 'validate_render',
                        'normal_{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx))
                else:
                    img_path = os.path.join(
                        self.base_exp_dir, 'normals',
                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx))

                cv.imwrite(img_path, normal_img[..., i])

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0, debug=False):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32) * self.dataset.world_scale
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32) * self.dataset.world_scale

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        # if world_space:
        #     vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        if not debug: mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        if debug:
            os.makedirs(os.path.join(self.base_exp_dir, 'validate_render'), exist_ok=True)
            mesh.export(os.path.join(self.base_exp_dir, 'validate_render', 'meshes_{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

