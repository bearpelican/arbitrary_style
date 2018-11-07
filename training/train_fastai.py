from fastai import *
from fastai.vision import *
torch.backends.cudnn.benchmark=True
import time

from adamw import AdamW
from scheduler import Scheduler, LRScheduler
from models import *
from loss import TransferLoss
from data import ContentStyleLoader, InputDataset, SimpleDataBunch
from dist import DDP, sum_tensor, reduce_tensor, env_world_size, env_rank

import torch.distributed.deprecated as dist
from torch.utils.data.distributed import DistributedSampler

import argparse

PATH = Path('data')

# Parsing
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#     parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--phases', type=str, help='Learning rate schedule')
    parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--print-freq', '-p', default=5, type=int, help='print every')
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
    print('Distributed initializing process group')
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=env_world_size())
    assert(env_world_size() == dist.get_world_size())
    print("Distributed: success (%d/%d)"%(args.local_rank, dist.get_world_size()))

# Batch size
# size,bs = 96,40
# size,bs = 128,36
size,bs = 256,20

data_norm,data_denorm = normalize_funcs(*imagenet_stats)

# Content Data
IMAGENET_PATH = PATH/'imagenet-sz/320/train'
imagenet_files = get_files(IMAGENET_PATH, recurse=True)

# COCO_PATH = PATH/'coco/resize'
# coco_files = get_files(COCO_PATH, recurse=True)

data_files = imagenet_files
cont_ds = InputDataset(data_files)

# Content Data
cont_tds = DatasetTfm(cont_ds, tfms=[crop_pad(size=size, is_random=False), flip_lr(p=0.5)], tfm_y=False, size=size, do_crop=True)
data_sampler = DistributedSampler(cont_tds, num_replicas=env_world_size(), rank=env_rank()) if is_distributed else None
cont_dl = DeviceDataLoader.create(cont_tds, tfms=data_norm, num_workers=8, 
                                   bs=bs, shuffle=(data_sampler is None), sampler=data_sampler)

# Style Data
STYLE_PATH_DTD = PATH/'style/dtd/images'
dtd_files = get_files(STYLE_PATH_DTD, recurse=True)

STYLE_PATH_PBN = PATH/'style/pbn/train'
pbn_files = get_files(STYLE_PATH_PBN, recurse=True)

style_ds = InputDataset(dtd_files+pbn_files)
style_tds = DatasetTfm(style_ds, tfms=[crop_pad(size=size, is_random=False), flip_lr(p=0.5)], tfm_y=False, size=size, do_crop=True)
style_dl = DeviceDataLoader.create(style_tds, tfms=data_norm, num_workers=8, bs=1, shuffle=True)

# Data loader
train_dl = ContentStyleLoader(cont_dl, style_dl)
data = SimpleDataBunch(train_dl, MODEL_PATH)

print('Loaded data')


# Callbacks
class DistributedRecorder(Recorder):
    def on_train_begin(self, pbar:PBar, metrics_names:Collection[str], **kwargs:Any)->None:
        self.b_count = 0
        super().on_train_begin(pbar, metrics_names, **kwargs)
        
    def on_backward_begin(self, last_loss:Tensor, smooth_loss:Tensor, **kwargs:Any)->None:
        self.b_count += 1
        if is_distributed:
            metrics = smooth_loss.clone().detach().float().cuda()
            smooth_loss = reduce_tensor(metrics).cpu().numpy()

            if self.b_count % 50 == 0:
                print('Losses:', smooth_loss)
            
        super().on_backward_begin(smooth_loss.sum())
        return last_loss.sum()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        if args.local_rank == 0:
            name = Path(args.save).stem
            print('Saving model:', name)
            self.learn.save(f'{name}_{epoch}')

@dataclass
class WeightScheduler(Callback):
    "Manage 1-Cycle style training as outlined in Leslie Smith's [paper](https://arxiv.org/pdf/1803.09820.pdf)."
    learn:Learner
    loss_func:TransferLoss
    cont_phases:Collection[Tuple]
    style_phases:Collection[Tuple]

    def steps(self, phases):
        "Build anneal schedule for all of the parameters."
        n_batch = len(self.learn.data.train_dl)
        return [Stepper((start,end),ep*n_batch,annealing_linear) for ep,start,end in phases]

    def on_train_begin(self, n_epochs:int, **kwargs:Any)->None:
        "Initialize our optimization params based on our annealing schedule."
        self.style_scheds = list(reversed(self.steps(self.style_phases)))
        self.cont_scheds = list(reversed(self.steps(self.cont_phases)))
        
        self.cur_style = self.style_scheds.pop()
        self.cur_cont = self.cont_scheds.pop()

    def on_batch_end(self, train, **kwargs:Any)->None:
        "Take one step forward on the annealing schedule for the optim params."
        if train:
            self.loss_func.cont_wgt = self.cur_cont.step()
            self.loss_func.style_wgt = self.cur_style.step()

            if self.cur_style.is_done: self.cur_style = self.style_scheds.pop()
            if self.cur_cont.is_done: self.cur_cont = self.cont_scheds.pop()


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
opt_func = partial(AdamW, betas=(0.9,0.999), weight_decay=1e-3)

st_wgt = 2.5e9
ct_wgt = 5e2
tva_wgt = 1e-6
st_block_wgts = [1,80,200,5] # 2,3,4,5
c_block = 1 # 1=3
lr_mult = env_world_size()

epochs = 10
style_phases = [(2,1e2,st_wgt*3),(epochs,st_wgt,st_wgt)]
cont_phases = [(2,ct_wgt,ct_wgt),(2,ct_wgt,ct_wgt*3),(epochs,ct_wgt,ct_wgt)]


loss_func = TransferLoss(m_vgg, ct_wgt, st_wgt, st_block_wgts, tva_wgt, data_norm, c_block)

learner = Learner(data, m_com, opt_func=opt_func, loss_func=loss_func)
w_sched = partial(WeightScheduler, loss_func=loss_func, cont_phases=cont_phases, style_phases=style_phases)
learner.callback_fns = [DistributedRecorder, w_sched]

print('Begin training')
learner.fit_one_cycle(epochs, 1e-5*lr_mult)

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
    