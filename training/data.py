from fastai import *
from fastai.vision import *

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