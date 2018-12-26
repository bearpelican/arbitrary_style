from fastai import *
from fastai.vision import *
torch.backends.cudnn.benchmark=True
import time

from adamw import AdamW
from scheduler import Scheduler, LRScheduler
from models import *
from loss import TransferLoss
from data import get_data, SimpleDataBunch
from dist import DDP, sum_tensor, reduce_tensor, env_world_size, env_rank
from callbacks import DistributedRecorder, WeightScheduler

import torch.distributed.deprecated as dist

import argparse

PATH = Path('data')

# Parsing
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#     parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--phases', type=str, help='Learning rate schedule')
    parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--print-freq', '-p', default=50, type=int, help='print every')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int, help='Used for multi-process training')
    # parser.add_argument('--load', action='store_true', help='Load model')
    parser.add_argument('--resnet', action='store_true', help='Use resnet arch')
    parser.add_argument('--load', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    return parser

args = get_parser().parse_args()


MODEL_PATH = Path(args.save).expanduser().parent

is_distributed = env_world_size() > 1
if args.local_rank > 0:
    f = open('/dev/null', 'w')
    sys.stdout = f

print('Starting script')

if is_distributed:
    torch.cuda.set_device(args.local_rank)
    print('Distributed initializing process group')
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=env_world_size())
    assert(env_world_size() == dist.get_world_size())
    print("Distributed: success (%d/%d)"%(args.local_rank, dist.get_world_size()))

# Batch size
# size,bs = 96,40
# size,bs = 128,36
size,bs = 256,20

# Content Data
IMAGENET_PATH = PATH/'imagenet-sz/320/train'
COCO_PATH = PATH/'coco/resize'
STYLE_PATH_DTD = PATH/'style/dtd/images'
STYLE_PATH_PBN = PATH/'style/pbn/train'

imagenet_files = get_files(IMAGENET_PATH, recurse=True)
# coco_files = get_files(COCO_PATH, recurse=True)
dtd_files = get_files(STYLE_PATH_DTD, recurse=True)
pbn_files = get_files(STYLE_PATH_PBN, recurse=True)

train_dl = get_data(imagenet_files, dtd_files+pbn_files, size=size, cont_bs=bs)
data = SimpleDataBunch(train_dl, MODEL_PATH)

print('Loaded data')

# Create models
mt = StyleTransformer()
ms = StylePredict.create_resnet() if args.resnet else StylePredict.create_inception()
m_com = CombinedModel(mt, ms).cuda()
if is_distributed: 
    m_com = DDP(m_com, device_ids=[args.local_rank], output_device=args.local_rank)

load_path = Path(args.load)
if args.load and load_path.exists(): 
    print('Loading model from path:', load_path)
    m_com.load_state_dict(torch.load(load_path.expanduser(), map_location=lambda storage,loc: storage.cuda(args.local_rank)), strict=True)
    
m_vgg = VGGActivations().cuda()

print('Created models')

# Training
opt_func = partial(optim.Adam, betas=(0.9,0.999), weight_decay=1e-3)

st_wgt = 4e9
ct_wgt = 5e2    
tva_wgt = 1e-6
st_block_wgts = [1,80,200,5] # 2,3,4,5
c_block = 1 # 1=3
lr_mult = env_world_size()

epochs = 20
# style_phases = [(1,1e2,st_wgt*2),(1,st_wgt,st_wgt/2)]*2 + [(epochs,st_wgt,st_wgt)]
# style_phases = [(1,st_wgt,st_wgt*2),(1,st_wgt,st_wgt)]*2 + [(epochs,st_wgt,st_wgt)]
# cont_phases = [(1,ct_wgt,ct_wgt/2),(1,ct_wgt,ct_wgt*2)]*2 + [(epochs,ct_wgt,ct_wgt)]
style_phases = [(epochs,st_wgt,st_wgt)]
cont_phases = [(epochs,ct_wgt,ct_wgt)]


loss_func = TransferLoss(m_vgg, ct_wgt, st_wgt, st_block_wgts, tva_wgt, c_block)

learner = Learner(data, m_com, opt_func=opt_func, loss_func=loss_func)
w_sched = partial(WeightScheduler, loss_func=loss_func, cont_phases=cont_phases, style_phases=style_phases)
recorder = partial(DistributedRecorder, save_path=args.save, print_freq=args.print_freq)
learner.callback_fns = [recorder, w_sched]

print('Begin training')
learner.fit_one_cycle(epochs, 5e-5*lr_mult)

def eval_imgs(x_con, x_style, idx=0):
    with torch.no_grad(): 
        out = m_com(x_con, x_style)
    fig, axs = plt.subplots(1,3,figsize=(12,4))
    Image(data_denorm(x_con[idx].cpu())).show(axs[0])
    axs[0].set_title('Content')
    axs[1].set_title('Style')
    axs[2].set_title('Transfer')
    Image(data_denorm(x_style[0].cpu())).show(axs[1])
    Image((out[idx].detach().cpu())).show(axs[2])
    