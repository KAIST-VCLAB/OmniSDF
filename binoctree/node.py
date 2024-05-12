import sys
import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Union

PI = float(np.pi)

def floating_eq(a, b):
    return np.all(np.abs(a-b) < 1e-5)

def ISO_Cartesian2Sphere(x, y, z):
    """     z
            |   /
            | theta
            |*/
            |/
            o----------y
            /\
           /**\
          /    phi 
        x       
    """
    r = np.sqrt(x**2 + y **2 + z**2)
    theta = np.arccos(z / r)
    if x != 0:
        phi = np.arctan2(y, x)
        if phi < 0:
            phi += 2*PI
    else:
        if y > 0:
            phi = PI/2
        elif y < 0:
            phi = PI*3/2
        else:
            return 0, 0, 0
    return r, theta, phi

def ISO_Sphere2Cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def torch_Sphere2Carteisian(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)
    

@dataclass
class sVertex:
    r: float
    theta: float
    phi: float
    idx: int
    def __post_init__(self):
        self.x, self.y, self.z = ISO_Sphere2Cartesian(self.r, self.theta, self.phi)

@dataclass
class Sphoxel:
    OCTREE: dataclass
    min_r: float
    max_r: float
    min_theta: float
    max_theta: float
    min_phi: float
    max_phi: float                      # Boundary
    v: Union[None, List[int]] = None    # Vertices index list
    parent: dataclass = None            # Parent Sphoxel object
    child: List[dataclass] = field(default_factory=list)    # Child Sphoxel object
    level: int = 0
    isleaf: bool = True
    active: bool = False
    idx: int = 0
    hit: int = 0
    sample: bool = False
    
    
    @torch.no_grad()
    def __post_init__(self):
        # Set R center
        mid_depth_rule = self.OCTREE.mid_depth_rule
        if mid_depth_rule =='am':     # arithmetic mean
            self.mid_r = (self.min_r+self.max_r)/2
        elif mid_depth_rule == 'gm':  # geometric mean
            self.mid_r = math.sqrt(self.min_r*self.max_r)
        elif mid_depth_rule == 'hm':  # harmonic mean
            self.mid_r = 2/((1/self.min_r+1/self.max_r))
        else:
            sys.exit()
        
        # Set Theta, Phi center
        self.mid_theta = (self.min_theta + self.max_theta) / 2
        self.mid_phi = (self.min_phi + self.max_phi) / 2
        
        self.idx = self.OCTREE.VOXEL_CNT
        self.OCTREE.VOXEL_CNT += 1
        self.OCTREE.SPHOXEL_LIST.append(self)
        self.sr = self.size()   # Solid angle [steradian]

        
    def collision(self, x, y, z):
        r, theta, phi = ISO_Cartesian2Sphere(x, y, z)
        flag = self.min_r <= r < self.max_r and \
            self.min_theta <= theta < self.max_theta and \
            self.min_phi <= phi < self.max_phi
        return flag
    
    def get_center(self):
        x, y, z = ISO_Sphere2Cartesian((self.min_r + self.max_r)/2, self.mid_theta, self.mid_phi)
        return np.array([x, y, z])
    
    def volume(self):
        volume = (self.max_r**3 - self.min_r**3) *  \
                (np.cos(self.min_theta) - np.cos(self.max_theta)) * \
                (self.max_phi - self.min_phi) / 3
        return volume

    def size(self):
        if self.idx == 0: # root
            return 4 * PI
        
        volume = (self.max_r**3 - self.min_r**3) *  \
                (np.cos(self.min_theta) - np.cos(self.max_theta)) * \
                (self.max_phi - self.min_phi) / 3
        radi = np.cbrt(3 * volume / (4 * PI))
        d = (self.min_r + self.max_r) / 2
        alpha = np.arcsin(radi / d)
        strd = 4 * PI * np.sin(alpha / 2) ** 2
        
        return strd
    
    def radius(self):
        volume = (self.max_r**3 - self.min_r**3) *  \
                (np.cos(self.min_theta) - np.cos(self.max_theta)) * \
                (self.max_phi - self.min_phi) / 3
        radi = np.cbrt(3 * volume / (4 * PI))
        return radi

    def get_cuboid_center(self):
        vertices = self.get_vertices_xyz()  #[8, 3]
        return np.mean(vertices, axis=0)
    
    def get_cuboid_distance(self):
        center = self.get_cuboid_center()
        vertices = self.get_vertices_xyz()  #[8, 3]
        radi = np.max(np.linalg.norm(vertices-center[None, :], axis=-1))
        return float(radi)

    
    def child_idx(self, r, theta, phi):
        if len(self.child) == 8:
            rdx = 0 if r < self.mid_r else 1
            tdx = 0 if theta < self.mid_theta else 1
            pdx = 0 if phi < self.mid_phi else 1
            idx = rdx * 4 + tdx * 2 + pdx 
        elif len(self.child) == 2:
            idx = 0 if r < self.mid_r else 1
            return idx
        else:
            print("Undefined argument in Sphoxel.child_idx()")
            sys.exit()
        return idx

    def add_vertex(self, r, theta, phi):
        vtx = sVertex(r, theta, phi, self.OCTREE.VERTEX_CNT)
        self.OCTREE.VERTEX_CNT += 1
        self.OCTREE.v.append(vtx)
        return vtx.idx

    def create_child_vertex(self):
        vidx = {}
        vidx[1] = self.add_vertex(self.min_r, self.mid_theta, self.min_phi)
        vidx[2] = self.add_vertex(self.min_r, self.mid_theta, self.mid_phi)
        vidx[3] = self.add_vertex(self.min_r, self.mid_theta, self.max_phi)

        vidx[13] = self.add_vertex(self.mid_r, self.mid_theta, self.min_phi)
        vidx[14] = self.add_vertex(self.mid_r, self.mid_theta, self.mid_phi)
        vidx[15] = self.add_vertex(self.mid_r, self.mid_theta, self.max_phi)

        vidx[21] = self.add_vertex(self.max_r, self.mid_theta, self.min_phi)
        vidx[22] = self.add_vertex(self.max_r, self.mid_theta, self.mid_phi)
        vidx[23] = self.add_vertex(self.max_r, self.mid_theta, self.max_phi)

        if floating_eq(self.min_theta, 0):
            vidx[0] = self.v[0]
            vidx[10] = self.add_vertex(self.mid_r, self.min_theta, self.mid_phi)
            vidx[11] = vidx[10]
            vidx[12] = vidx[10]
            vidx[20] = self.v[4]

            vidx[4] = self.add_vertex(self.min_r, self.max_theta, self.mid_phi)
            vidx[16] = self.add_vertex(self.mid_r, self.max_theta, self.min_phi)
            vidx[17] = self.add_vertex(self.mid_r, self.max_theta, self.mid_phi)
            vidx[18] = self.add_vertex(self.mid_r, self.max_theta, self.max_phi)
            vidx[24] = self.add_vertex(self.max_r, self.max_theta, self.mid_phi)

        elif floating_eq(self.max_theta, PI):
            vidx[0] = self.add_vertex(self.min_r, self.min_theta, self.mid_phi)
            vidx[10] = self.add_vertex(self.mid_r, self.min_theta, self.min_phi)
            vidx[11] = self.add_vertex(self.mid_r, self.min_theta, self.mid_phi)
            vidx[12] = self.add_vertex(self.mid_r, self.min_theta, self.max_phi)
            vidx[20] = self.add_vertex(self.max_r, self.min_theta, self.mid_phi)

            vidx[4] = self.v[2]
            vidx[16] = self.add_vertex(self.mid_r, self.max_theta, self.mid_phi)
            vidx[17] = vidx[16]
            vidx[18] = vidx[16]
            vidx[24] = self.v[6]
        
        else:
            vidx[0] = self.add_vertex(self.min_r, self.min_theta, self.mid_phi)
            vidx[10] = self.add_vertex(self.mid_r, self.min_theta, self.min_phi)
            vidx[11] = self.add_vertex(self.mid_r, self.min_theta, self.mid_phi)
            vidx[12] = self.add_vertex(self.mid_r, self.min_theta, self.max_phi)
            vidx[20] = self.add_vertex(self.max_r, self.min_theta, self.mid_phi)

            vidx[4] = self.add_vertex(self.min_r, self.max_theta, self.mid_phi)
            vidx[16] = self.add_vertex(self.mid_r, self.max_theta, self.min_phi)
            vidx[17] = self.add_vertex(self.mid_r, self.max_theta, self.mid_phi)
            vidx[18] = self.add_vertex(self.mid_r, self.max_theta, self.max_phi)
            vidx[24] = self.add_vertex(self.max_r, self.max_theta, self.mid_phi)
        
        return vidx

    def create_binary_child_vertex(self):
        vidx = {}
        if floating_eq(self.min_theta, 0):
            vidx[0] = self.add_vertex(self.mid_r, self.min_theta, self.min_phi)
            vidx[1] = vidx[0]
            vidx[2] = self.add_vertex(self.mid_r, self.max_theta, self.min_phi)
            vidx[3] = self.add_vertex(self.mid_r, self.max_theta, self.max_phi)
        elif floating_eq(self.max_theta, PI):
            vidx[0] = self.add_vertex(self.mid_r, self.min_theta, self.min_phi)
            vidx[1] = self.add_vertex(self.mid_r, self.min_theta, self.max_phi)
            vidx[2] = self.add_vertex(self.mid_r, self.max_theta, self.min_phi)
            vidx[3] = vidx[2]
        else:
            vidx[0] = self.add_vertex(self.mid_r, self.min_theta, self.min_phi)
            vidx[1] = self.add_vertex(self.mid_r, self.min_theta, self.max_phi)
            vidx[2] = self.add_vertex(self.mid_r, self.max_theta, self.min_phi)
            vidx[3] = self.add_vertex(self.mid_r, self.max_theta, self.max_phi)
        return vidx

    def division_check(self):
        if self.OCTREE.subdivision_rule == 'size':
            return self.sr > self.OCTREE.max_voxel_size
        elif self.OCTREE.subdivision_rule == 'level':
            return self.level < self.OCTREE.max_tree_level
        elif self.OCTREE.subdivision_rule == 'both':
            if self.sr > self.OCTREE.max_voxel_size:
                if self.level < self.OCTREE.max_tree_level:
                    return True
            return False
        else:
            return False

    def elongated(self):
        return (1.4*(self.max_phi-self.min_phi)*(self.min_r + self.max_r)/2) < (self.max_r - self.min_r)
    
    def create_subtree(self, active=False):
        if not self.isleaf:
            return False
        else:
            self.isleaf=False
            
        if self.OCTREE.binary_division and self.elongated():
            chidx = self.create_binary_child_vertex()
            self.child = [
                Sphoxel(
                    self.OCTREE,
                    self.min_r, self.mid_r, self.min_theta, self.max_theta, self.min_phi, self.max_phi,
                    [self.v[0], self.v[1], self.v[2], self.v[3], chidx[0], chidx[1], chidx[2], chidx[3]],
                    child=[], parent=self, level=self.level+1,
                    isleaf=True, active=active),
                Sphoxel(
                    self.OCTREE,
                    self.mid_r, self.max_r, self.min_theta, self.max_theta, self.min_phi, self.max_phi,
                    [chidx[0], chidx[1], chidx[2], chidx[3], self.v[4], self.v[5], self.v[6], self.v[7]],
                    child=[], parent=self, level=self.level+1,
                    isleaf=True, active=active)
            ]
            return 2, chidx
        else:
            # Octave childs
            chidx = self.create_child_vertex()  # dict
            self.child = [
                Sphoxel( self.OCTREE, # 0
                    self.min_r, self.mid_r, self.min_theta, self.mid_theta, self.min_phi, self.mid_phi,
                    [self.v[0], chidx[0], chidx[1], chidx[2], chidx[10], chidx[11], chidx[13], chidx[14]],
                    parent=self, child=[], level=self.level + 1, isleaf=True, active=active),
                Sphoxel( self.OCTREE, # 1
                    self.min_r, self.mid_r, self.min_theta, self.mid_theta, self.mid_phi, self.max_phi,
                    [chidx[0], self.v[1], chidx[2], chidx[3], chidx[11], chidx[12], chidx[14], chidx[15]],
                    parent=self, child=[], level=self.level + 1, isleaf=True, active=active),
                Sphoxel( self.OCTREE, # 2
                    self.min_r, self.mid_r, self.mid_theta, self.max_theta, self.min_phi, self.mid_phi,
                    [chidx[1], chidx[2], self.v[2], chidx[4], chidx[13], chidx[14], chidx[16], chidx[17]],
                    parent=self, child=[], level=self.level + 1, isleaf=True, active=active),
                Sphoxel( self.OCTREE, # 3
                    self.min_r, self.mid_r, self.mid_theta, self.max_theta, self.mid_phi, self.max_phi,
                    [chidx[2], chidx[3], chidx[4], self.v[3], chidx[14], chidx[15], chidx[17], chidx[18]],
                    parent=self, child=[], level=self.level + 1, isleaf=True, active=active),
                Sphoxel( self.OCTREE, # 4
                    self.mid_r, self.max_r, self.min_theta, self.mid_theta, self.min_phi, self.mid_phi,
                    [chidx[10], chidx[11], chidx[13], chidx[14], self.v[4], chidx[20], chidx[21], chidx[22]],
                    parent=self, child=[], level=self.level + 1, isleaf=True, active=active),
                Sphoxel( self.OCTREE, # 5
                    self.mid_r, self.max_r, self.min_theta, self.mid_theta, self.mid_phi, self.max_phi,
                    [chidx[11], chidx[12], chidx[14], chidx[15], chidx[20], self.v[5], chidx[22], chidx[23]],
                    parent=self, child=[], level=self.level + 1, isleaf=True, active=active),
                Sphoxel( self.OCTREE, # 6
                    self.mid_r, self.max_r, self.mid_theta, self.max_theta, self.min_phi, self.mid_phi,
                    [chidx[13], chidx[14], chidx[16], chidx[17], chidx[21], chidx[22], self.v[6], chidx[24]],
                    parent=self, child=[], level=self.level + 1, isleaf=True, active=active),
                Sphoxel( self.OCTREE, # 7
                    self.mid_r, self.max_r, self.mid_theta, self.max_theta, self.mid_phi, self.max_phi,
                    [chidx[14], chidx[15], chidx[17], chidx[18], chidx[22], chidx[23], chidx[24], self.v[7]],
                    parent=self, child=[], level=self.level + 1, isleaf=True, active=active),
            ]
            return 8, chidx

    def subdivide(self, x, y, z, color=None):
        # Assume, xyz is always in self
        if self.level == 0:
            # Case Exception for ROOT node
            for node in self.child:
                # All subdivision runs if xyz included in self sphoxel
                if node.collision(x, y, z):
                    node.subdivide(x, y, z, color)
        else:
            if self.division_check():
                r, theta, phi = ISO_Cartesian2Sphere(x, y, z)
                self.create_subtree()
                cidx = self.child_idx(r, theta, phi) # Get child that intersects [x, y, z]
                self.child[cidx].subdivide(x, y, z, color)
            else:
                self.hit += 1

    def get_spherical_bound(self):
        return np.array([self.min_r, self.max_r, self.min_theta, self.max_theta, self.min_phi, self.max_phi])

    def get_vertices_xyz(self):
        result = []
        for vidx in self.v:
            vobj = self.OCTREE.v[vidx]
            result.append(np.array([vobj.x, vobj.y, vobj.z]))
        return np.stack(result, axis=0)

    def get_child_idx(self):
        result = []
        for sphoxel in self.child:
            result.append(sphoxel.idx)
        return np.array(result, np.int32)
