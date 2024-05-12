from time import time
from dataclasses import dataclass
import torch

@dataclass
class Timer:
    flag = 0
    t1: float = 0
    t2: float = 0
    cnt: int =0

    def __post_init__(self):
        self.interval = {}
        self.stack_t1 = {}
        self.stack_t2 = {}
        self.stack_flag = {}
        self.stack_cnt = {}
    
    
    def tick(self, msg=None):
        if self.flag == 0:
            torch.cuda.synchronize()
            self.t1 = time()
            self.flag += 1
        elif self.flag == 1:
            torch.cuda.synchronize()
            self.t2 = time()
            self.flag = 0
            print(f'{msg}\t:\t{self.t2-self.t1:.02f} seconds')
        else:
            print("Timer error")

    def stack_tick(self, key=None):
        if key not in self.interval.keys():
            self.interval[key] = 0
            self.stack_t1[key] = 0
            self.stack_t2[key] = 0
            self.stack_flag[key] = 0
            self.stack_cnt[key] = 0

        if self.stack_flag[key] == 0:
            self.stack_t1[key] = time()
            self.stack_flag[key] += 1
            
        elif self.stack_flag[key] == 1:
            self.stack_t2[key] = time()
            self.stack_flag[key] = 0
            
            self.interval[key] = ((self.interval[key] * self.stack_cnt[key]) \
                + (self.stack_t2[key]-self.stack_t1[key])) / (self.stack_cnt[key] + 1)
            self.stack_cnt[key] += 1
        else:
            print("Timer error")
    def print_stacks(self):
        for key in self.interval.keys():
            print(f'{key}\t:\t{self.interval[key]:.05f} seconds')

            