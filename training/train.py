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

IMAGENET_PATH = PATH/'imagenet-sz/320/train'
train_ds = ImageClassificationDataset.from_folder(IMAGENET_PATH)

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
train_dl = ContentStyleLoader(cont_dl, style_dl, repeat_xy=False)

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
    
# Training
epochs = 10
optimizer = AdamW(m_com.parameters(), lr=1e-5, betas=(0.9,0.999), weight_decay=1e-5)

lr_mult = env_world_size()
scheduler = LRScheduler(optimizer, [{'ep': (0,3),      'lr': (1e-5*lr_mult,5e-4*lr_mult)}, 
                                    {'ep': (3,6),      'lr': (5e-4*lr_mult,1e-5*lr_mult)},
                                    {'ep': (6,epochs), 'lr': (1e-5*lr_mult,1e-7*lr_mult)}])

st_wgt = 2.5e9
# st_scheduler = Scheduler([{'ep': (0,1), 'st': (st_wgt)}, 
st_scheduler = Scheduler([{'ep': (0,2), 'st': (st_wgt if args.load else 1e2,st_wgt*4)}, 
                          {'ep': 2,     'st': (st_wgt)}], 'st')
ct_wgt = 5e2
# st_wgt = 1e8
tva_wgt = 1e-6
style_block_wgts = [1,80,200,5] # 2,3,4,5
c_block = 1 # 1=3


m_vgg = VGGActivations().cuda()
m_loss = TransferLoss(m_vgg, ct_wgt, st_wgt, st_block_wgts, tva_wgt, data_norm, c_block)



start = time.time()
m_com.train()
style_image_count = 0
for e in range(epochs):
    agg_content_loss = 0.
    agg_style_loss = 0.
    agg_tva_loss = 0.
    count = 0
    batch_tot = len(train_dl)
    for batch_id, (x_con,x_style) in enumerate(train_dl):
        scheduler.update_lr(e, batch_id, batch_tot)
            
        n_batch = x_con.size(0)
        count += n_batch
        optimizer.zero_grad()
        
        out = m_com(x_con, x_style)
        out,_ = data_norm((out,None))
        
        m_loss.style_wgt = st_scheduler.get_val(e, batch_id, batch_tot)
        total_loss = m_loss(out, x_con, x_style)
        closs, sloss, tvaloss = total_loss.copy().detach().cpu()

        total_loss.sum().backward()
        m_clip = m_com.module if is_distributed else m_com
        nn.utils.clip_grad_norm_(m_clip.m_tran.parameters(), 10)
        nn.utils.clip_grad_norm_(m_clip.m_style.parameters(), 100)
        optimizer.step()
    
        mom = 0.9
        agg_content_loss = agg_content_loss*mom + closs*(1-mom)
        agg_style_loss = agg_style_loss*mom + sloss*(1-mom)
        agg_tva_loss = agg_tva_loss*mom + tvaloss*(1-mom)
        agg_total_loss = (agg_content_loss + agg_style_loss + agg_tva_loss)

        if is_distributed: # Must keep track of global batch size, since not all machines are guaranteed equal batches at the end of an epoch
            metrics = torch.tensor([agg_content_loss, agg_style_loss, agg_tva_loss, agg_total_loss]).float().cuda()
            agg_content_loss, agg_style_loss, agg_tva_loss, agg_total_loss = reduce_tensor(metrics).cpu().numpy()
        
        if (batch_id + 1) % args.print_freq == 0:
            time_elapsed = (time.time() - start)/60
            mesg = (f"MIN:{time_elapsed:.2f}\tEP[{e+1}]\tB[{batch_id+1:4}/{batch_tot}]\t"
                    f"CON:{agg_content_loss:.3f}\tSTYL:{agg_style_loss:.2f}\t"
                    f"TVA:{agg_tva_loss:.2f}\tTOT:{agg_total_loss:.2f}\t"
                    f"S/CT:{style_image_count:3}/{count:3}"
                   )
            print(mesg)

        save_interval = 1000
        if (args.local_rank == 0) and (batch_id+1) % save_interval == 0:
            if args.save:
                print('Saving model: ', args.save)
                save_path = Path(args.save).expanduser()
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(m_com.state_dict(), save_path)

                ep_save = save_path.with_name(f'{save_path.stem}_{e}').with_suffix(save_path.suffix)

                print('Saving epoch checkpoint: ', ep_save)
                torch.save(m_com.state_dict(), ep_save)

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
    