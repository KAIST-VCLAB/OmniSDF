import sys
import numpy as np
import torch
from binoctree.node import sVertex, Sphoxel, PI, torch_Sphere2Carteisian
from utils.metric import Timer
timer = Timer()

def floating_eq(a, b):
    return np.all(np.abs(a-b) < 1e-5)

class Octree:
    def __init__(self,
                 near,
                 far,
                 max_size=0.0001,
                 max_level=-1,
                 mid_depth_rule='am',
                 subdivision_rule='size',
                 binary_division=True,
                 active_height = 0
                 ):
        
        self.SPHOXEL_LIST = []
        self.near = float(near)
        self.far = float(far)
        self.max_voxel_size = max_size
        self.max_tree_level = max_level
        self.VOXEL_CNT = 0
        self.VERTEX_CNT = 13
        self.mid_depth_rule = mid_depth_rule
        self.subdivision_rule = subdivision_rule
        self.binary_division = binary_division
        self.active_height = active_height
        self.leaf_vec = []


        self.v = [
            sVertex(0, 0, 0, 0),                 # Add default vertex
            sVertex(self.near, 0,    0,      1), # phi (PI/2, PI, 3PI/2, 2PI)
            sVertex(self.near, PI/2, 0,      2),
            sVertex(self.near, PI/2, PI/2,   3),
            sVertex(self.near, PI/2, PI,     4),
            sVertex(self.near, PI/2, PI*3/2, 5),
            sVertex(self.near, PI,   0,      6),
            sVertex(self.far, 0,     0,      7), # phi (PI/2, PI, 3PI/2, 2PI)
            sVertex(self.far, PI/2,  0,      8),
            sVertex(self.far, PI/2,  PI/2,   9),
            sVertex(self.far, PI/2,  PI,     10),
            sVertex(self.far, PI/2,  PI*3/2, 11),
            sVertex(self.far, PI,    0,      12),
        ]

        self.root = Sphoxel(
            self, self.near, self.far, 0, PI, 0, 2*PI,
            v=[0,0,0,0,0,0,0,0], child=None, parent=None, level=0,
            isleaf=False, active=False
        )

        root_child = [   # r, theta, phi
            Sphoxel(self, self.near, self.far, 0, PI/2,   0,      PI/2,   [1, 1, 2, 3, 7, 7, 8, 9], child=[], parent=self.root, level=1),
            Sphoxel(self, self.near, self.far, 0, PI/2,   PI/2,   PI,     [1, 1, 3, 4, 7, 7, 9, 10], child=[], parent=self.root, level=1),
            Sphoxel(self, self.near, self.far, 0, PI/2,   PI,     PI*3/2, [1, 1, 4, 5, 7, 7, 10, 11], child=[], parent=self.root, level=1),
            Sphoxel(self, self.near, self.far, 0, PI/2,   PI*3/2, PI*2,   [1, 1, 5, 2, 7, 7, 11, 8], child=[], parent=self.root, level=1),
            Sphoxel(self, self.near, self.far, PI/2, PI,  0,      PI/2,   [2, 3, 6, 6, 8, 9, 12, 12], child=[], parent=self.root, level=1),
            Sphoxel(self, self.near, self.far, PI/2, PI,  PI/2,   PI,     [3, 4, 6, 6, 9, 10, 12, 12], child=[], parent=self.root, level=1),
            Sphoxel(self, self.near, self.far, PI/2, PI,  PI,    PI*3/2,  [4, 5, 6, 6, 10, 11, 12, 12], child=[], parent=self.root, level=1),
            Sphoxel(self, self.near, self.far, PI/2, PI,  PI*3/2, PI*2,   [5, 2, 6, 6, 11, 8, 12, 12], child=[], parent=self.root, level=1)
        ]
        self.root.child=root_child
    
    def compute_height(self, node):
        if node.isleaf:
            node.height = 0
            return 0
        else:
            heights = []
            for child in node.child:
                heights.append(self.compute_height(child))
            node.height = max(heights) + 1
            return node.height

    def count_voxels(self):
        tmp = torch.zeros(len(self.v), dtype=torch.int)
        n_voxels = 0
        for vox in self.SPHOXEL_LIST:
            if vox is not None:
                n_voxels += 1
                tmp[torch.tensor(vox.v)] = 1
        n_vertices = int(torch.sum(tmp).item())

        return n_voxels, n_vertices

    def clear_leaf(self):
        self.leaf_vec = []

    
    def collect_leaf(self, node, level):
        if node.isleaf:
            self.leaf_vec.append(node)
        else:
            if len(node.child) == 8:
                self.collect_leaf(node.child[3], level+1)
                self.collect_leaf(node.child[2], level+1)
                self.collect_leaf(node.child[1], level+1)
                self.collect_leaf(node.child[0], level+1)
                self.collect_leaf(node.child[7], level+1)
                self.collect_leaf(node.child[6], level+1)
                self.collect_leaf(node.child[5], level+1)
                self.collect_leaf(node.child[4], level+1)

            if len(node.child) == 2:
                self.collect_leaf(node.child[0], level+1)
                self.collect_leaf(node.child[1], level+1)
    

    def collect_info(self, node):
        self.leaf_vec.append(node)
        if node.volume() < self.min_sphoxel_size:
            self.min_sphoxel_size = node.volume()
        if node.active:
            if self.active_max_level < node.level:
                self.active_max_level = node.level
        if node.isleaf:
            if self.counted_max_level < node.level:
                self.counted_max_level = node.level
            return
            self.leaf_vec.append(node)
        else:
            for ch in node.child:
                self.collect_info(ch)

    @torch.no_grad()
    def pack_tree_tensor(self):
        timer.tick()
        print(self.VOXEL_CNT)
        specs = torch.zeros([self.VOXEL_CNT, 6], device="cuda:0",
                            dtype=torch.float32, requires_grad=False)
        childs = torch.zeros([self.VOXEL_CNT, 9], device="cuda:0", 
                            dtype=torch.int, requires_grad=False)
        
        for vox in self.SPHOXEL_LIST:
            if vox is not None:
                specs[vox.idx, :] = torch.as_tensor(
                    [vox.min_r, vox.max_r,
                    vox.min_theta, vox.max_theta,
                    vox.min_phi, vox.max_phi],
                    device="cuda:0")
                try:
                    childs[vox.idx, -1] = len(vox.child)
                    for i in range(len(vox.child)):
                        childs[vox.idx, i] = vox.child[i].idx
                except:
                    print(vox)
                    print(vox.isleaf)
                    print(vox.level)
                    print(vox.idx)
                    sys.exit()
                
        timer.tick("Collect voxels")

        return specs, childs
    
    @torch.no_grad()
    def compute_node_vertice(self, specs):
        timer.tick()
        v0 = torch_Sphere2Carteisian(specs[:, 0], specs[:, 2], specs[:, 4]) # [N, 3]
        v1 = torch_Sphere2Carteisian(specs[:, 0], specs[:, 2], specs[:, 5])
        v2 = torch_Sphere2Carteisian(specs[:, 0], specs[:, 3], specs[:, 4])
        v3 = torch_Sphere2Carteisian(specs[:, 0], specs[:, 3], specs[:, 5])
        v4 = torch_Sphere2Carteisian(specs[:, 1], specs[:, 2], specs[:, 4])
        v5 = torch_Sphere2Carteisian(specs[:, 1], specs[:, 2], specs[:, 5])
        v6 = torch_Sphere2Carteisian(specs[:, 1], specs[:, 3], specs[:, 4])
        v7 = torch_Sphere2Carteisian(specs[:, 1], specs[:, 3], specs[:, 5])
        vertices = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=1)   # [N, 8, 3]
        timer.tick("Compute vertices")
        return vertices

    @torch.no_grad()
    def compute_node_size(self, vertices):
        timer.tick()
        center = torch.mean(vertices, dim=1, keepdims=False) #[N, 3]
        radius = torch.max(torch.linalg.norm(vertices - center[:, None, :], dim=-1), dim=-1)[0]
        timer.tick("Compute center and radius")
        return center, radius
    
    
    def collect_active(self, node, collect=False):
        if collect:
            self.leaf_vec.append(node)
        else:
            if node.active:
                self.leaf_vec.append(node)
                collect=True
                
            if len(node.child) == 8:
                self.collect_active(node.child[3], collect)
                self.collect_active(node.child[2], collect)
                self.collect_active(node.child[1], collect)
                self.collect_active(node.child[0], collect)
                self.collect_active(node.child[7], collect)
                self.collect_active(node.child[6], collect)
                self.collect_active(node.child[5], collect)
                self.collect_active(node.child[4], collect)

            if len(node.child) == 2:
                self.collect_active(node.child[0], collect)
                self.collect_active(node.child[1], collect)
        
        
    def serialize_leaf(self, kwds):
        sphoxel_obj = []
        sphoxel_idx = []
        sphoxel_hit = []
        sphoxel_bnd = []
        sphoxel_centers = []
        sphoxel_vidx = []
        sphoxel_chidx = []
        sphoxel_vxyz = []
        for sphoxel in self.SPHOXEL_LIST:
            if sphoxel.isleaf:        
                if "obj" in kwds: sphoxel_obj.append(sphoxel)
                if "voxel_idx" in kwds: sphoxel_idx.append(sphoxel.idx)
                if "voxel_hit" in kwds: sphoxel_hit.append(sphoxel.hit)
                if "voxel_range" in kwds: sphoxel_bnd.append(sphoxel.get_spherical_bound())
                if "voxel_center" in kwds: sphoxel_centers.append(sphoxel.get_center())
                if "vertex_idx" in kwds: sphoxel_vidx.append(sphoxel.v)
                if "child_idx" in kwds: sphoxel_chidx.append(sphoxel.get_child_idx())
                if "vertex_xyz" in kwds: sphoxel_vxyz.append(sphoxel.get_vertices_xyz())
        
        output_dict = {}
        if "obj" in kwds: output_dict["voxel_obj"] = sphoxel_obj
        if "voxel_idx" in kwds: output_dict["voxel_idx"] = np.array(sphoxel_idx, dtype=np.int32)
        if "voxel_hit" in kwds: output_dict["voxel_hit"] = np.array(sphoxel_hit, dtype=np.int32)
        if "voxel_range" in kwds: output_dict["voxel_range"] = np.stack(sphoxel_bnd, axis=0)
        if "voxel_center" in kwds: output_dict["voxel_center"] = np.stack(sphoxel_centers, axis=0)
        if "vertex_idx" in kwds: output_dict["vertex_idx"] = np.array(sphoxel_vidx, dtype=np.int32)
        if "child_idx" in kwds: output_dict["child_idx"] = np.stack(sphoxel_chidx, axis=0)
        if "vertex_xyz" in kwds: output_dict["vertex_xyz"] = np.stack(sphoxel_vxyz, axis=0)
        
        return output_dict
    
    def print_child_status(self):
        cnt0, cnt2, cnt8, cntErr = 0, 0, 0, 0
        for obj in self.SPHOXEL_LIST:
            if len(obj.child) == 2:
                cnt2 += 1
            elif len(obj.child) == 8:
                cnt8 += 1
            elif len(obj.child) == 0:
                cnt0 += 1
            else:
                cntErr += 1
        print(f"Voxel count of (0 child, 2 child, 8 child, etc) = {cnt0}, {cnt2}, {cnt8}, {cntErr}")
    
    def get_max_level(self, node):
        if node.isleaf:
            return node.level
        else:
            ls = [self.get_max_level(voxel) for voxel in node.child]
            return max(ls)
    

    def collect_vertex(self, node):
        if node.isleaf:
            return
        else:
            if len(node.child) == 8:
                i_bf = len(self.v)
                chidx = node.create_child_vertex()
                i_af = len(self.v)

                i_arr = [0] * (i_af - i_bf)
                i_arr[0] = node.child[0].v[2]   # 1
                i_arr[1] = node.child[0].v[3]   # 2
                i_arr[2] = node.child[1].v[3]   # 3
                i_arr[3] = node.child[2].v[4]   # 13
                i_arr[4] = node.child[2].v[5]   # 14
                i_arr[5] = node.child[3].v[5]   # 15
                i_arr[6] = node.child[4].v[6]   # 21
                i_arr[7] = node.child[4].v[7]   # 22
                i_arr[8] = node.child[5].v[7]   # 23

                if floating_eq(node.min_theta, 0):
                    i_arr[9] = node.child[0].v[4]   # 10
                    i_arr[10] = node.child[2].v[3]  # 4
                    i_arr[11] = node.child[2].v[6]  # 16
                    i_arr[12] = node.child[2].v[7]  # 17
                    i_arr[13] = node.child[3].v[7]  # 18
                    i_arr[14] = node.child[6].v[7]  # 24
                elif floating_eq(node.max_theta, PI):
                    i_arr[9] = node.child[0].v[2]   # 0
                    i_arr[10] = node.child[0].v[4]  # 10
                    i_arr[11] = node.child[0].v[5]  # 11
                    i_arr[12] = node.child[1].v[5]  # 12
                    i_arr[13] = node.child[4].v[5]  # 20
                    i_arr[14] = node.child[2].v[6]  # 16
                else:
                    i_arr[9] = node.child[0].v[1]   # 0
                    i_arr[10] = node.child[0].v[4]  # 10
                    i_arr[11] = node.child[0].v[5]  # 11
                    i_arr[12] = node.child[1].v[5]  # 12
                    i_arr[13] = node.child[4].v[5]  # 20
                    i_arr[14] = node.child[2].v[3]  # 4
                    i_arr[15] = node.child[2].v[6]  # 16
                    i_arr[16] = node.child[2].v[7]  # 17
                    i_arr[17] = node.child[3].v[7]  # 18
                    i_arr[18] = node.child[6].v[7]  # 24

                self.v_map.extend(i_arr)
                #                   0           1         2         3          4           5          6         7  
                node.child[0].v = [node.v[0], chidx[0], chidx[1], chidx[2], chidx[10], chidx[11], chidx[13], chidx[14]]
                node.child[1].v = [chidx[0], node.v[1], chidx[2], chidx[3], chidx[11], chidx[12], chidx[14], chidx[15]]
                node.child[2].v = [chidx[1], chidx[2], node.v[2], chidx[4], chidx[13], chidx[14], chidx[16], chidx[17]]
                node.child[3].v = [chidx[2], chidx[3], chidx[4], node.v[3], chidx[14], chidx[15], chidx[17], chidx[18]]
                node.child[4].v = [chidx[10], chidx[11], chidx[13], chidx[14], node.v[4], chidx[20], chidx[21], chidx[22]]
                node.child[5].v = [chidx[11], chidx[12], chidx[14], chidx[15], chidx[20], node.v[5], chidx[22], chidx[23]]
                node.child[6].v = [chidx[13], chidx[14], chidx[16], chidx[17], chidx[21], chidx[22], node.v[6], chidx[24]]
                node.child[7].v = [chidx[14], chidx[15], chidx[17], chidx[18], chidx[22], chidx[23], chidx[24], node.v[7]]
            
            if len(node.child) == 2:
                i_bf = len(self.v)
                chidx = node.create_binary_child_vertex()
                i_af = len(self.v)
                i_arr = [0] * (i_af - i_bf)
                if floating_eq(node.min_theta, 0):
                    i_arr[0] = node.child[1].v[0]   # 0
                    i_arr[1] = node.child[1].v[2]   # 2
                    i_arr[2] = node.child[1].v[3]   # 3
                elif floating_eq(node.max_theta, PI):
                    i_arr[0] = node.child[1].v[0]   # 0
                    i_arr[1] = node.child[1].v[1]   # 1
                    i_arr[2] = node.child[1].v[2]   # 2
                else:
                    i_arr[0] = node.child[1].v[0]   # 0
                    i_arr[1] = node.child[1].v[1]   # 1
                    i_arr[2] = node.child[1].v[2]   # 2
                    i_arr[3] = node.child[1].v[3]   # 3
                self.v_map.extend(i_arr)
                #                   0           1         2         3          4           5          6         7  
                node.child[0].v = [node.v[0], node.v[1], node.v[2], node.v[3], chidx[0], chidx[1], chidx[2], chidx[3]]
                node.child[1].v = [chidx[0], chidx[1], chidx[2], chidx[3], node.v[4], node.v[5], node.v[6], node.v[7]]

            for ch in node.child:
                self.collect_vertex(ch)

    def collect_voxel(self, node):
        node.idx = self.VOXEL_CNT
        self.VOXEL_CNT += 1
        self.SPHOXEL_LIST.append(node)

        if node.isleaf:
            return
        else:
            for ch in node.child:
                self.collect_voxel(ch)

    def reorganize_tree(self):
        self.SPHOXEL_LIST = []
        self.VOXEL_CNT = 0
        self.collect_voxel(self.root)

        self.v = self.v[:13]
        self.VERTEX_CNT = 13
        
        self.v_map = []
        for voxel in self.root.child:
            self.collect_vertex(voxel)
        return self.v_map

        
    def set_active_height(self, height):
        for voxel in self.SPHOXEL_LIST:
            if not voxel.isleaf:
                voxel.hit = 0
            voxel.active=False
        
        for voxel in self.SPHOXEL_LIST:
            if voxel.hit > 0:
                active_voxel = voxel
                for step in range(height):
                    active_voxel = active_voxel.parent
                active_voxel.active = True
                voxel.active=False
        self.active_height = height
