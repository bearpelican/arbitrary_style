from fastai import *
from fastai.vision import *
from dist import env_world_size, env_rank
from torch.utils.data.distributed import DistributedSampler

def get_data(content_files, style_files, size=256, cont_bs=20):
    data_norm,data_denorm = normalize_funcs(*imagenet_stats)

    cont_ds = InputDataset(content_files)

    # Content Data
    cont_tds = DatasetTfm(cont_ds, tfms=[crop_pad(size=size, is_random=False), flip_lr(p=0.5)], tfm_y=False, size=size, do_crop=True)
    data_sampler = DistributedSampler(cont_tds, num_replicas=env_world_size(), rank=env_rank()) if env_world_size() > 1 else None
    cont_dl = DeviceDataLoader.create(cont_tds, tfms=data_norm, num_workers=8, 
                                    bs=cont_bs, shuffle=(data_sampler is None), sampler=data_sampler)

    style_ds = InputDataset(style_files)
    style_tds = DatasetTfm(style_ds, tfms=[crop_pad(size=size, is_random=False), flip_lr(p=0.5)], tfm_y=False, size=size, do_crop=True)
    style_dl = DeviceDataLoader.create(style_tds, tfms=data_norm, num_workers=8, bs=1, shuffle=True)

    # Data loader
    return ContentStyleLoader(cont_dl, style_dl)


class InputDataset(Dataset):
    def __init__(self, fns:FilePathList):
        super().__init__()
        self.x  = np.array(fns)
        self.image_opener = open_image

    def __getitem__(self, i): return self.image_opener(self.x[i]),0    
    def __len__(self): return len(self.x)
    
class ContentStyleLoader():
    def __init__(self, cont_dl, style_dl, repeat_xy=True):
        self.cont_dl,self.style_dl = cont_dl,style_dl
        self.repeat_xy=repeat_xy
        self.count_c = 0
        self.count_s = 0
        
    def __iter__(self):
        it_c = iter(self.cont_dl)
        it_s = iter(self.style_dl)
        
        for b_c in it_c:
            self.count_c += 1
            
            try: 
                b_s = next(it_s)
                self.count_s += 1
            except: 
                print('Restarting style')
                it_s = iter(self.style_dl)
                
            out = b_c[0], b_s[0]
            if self.repeat_xy: yield out, out
            else: yield out
        
    def __len__(self):
        return len(self.cont_dl)
    
class SimpleDataBunch():
    def __init__(self, train_dl, path): 
        self.train_dl,self.path = train_dl,path
        self.device,self.loss_func = defaults.device,None
        self.valid_dl = None