import torch.distributed.deprecated as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.deprecated import DistributedDataParallel
import os


# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel


class DDP(DistributedDataParallel):
    def load_state_dict(self, *args, **kwargs): self.module.load_state_dict(*args, **kwargs)
    def state_dict(self, *args, **kwargs): return self.module.state_dict(*args, **kwargs)

def reduce_tensor(tensor): return sum_tensor(tensor)/env_world_size()
def sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    return rt
def env_world_size(): return int(os.environ.get('WORLD_SIZE', 1))
def env_rank(): return int(os.environ.get('RANK',0))