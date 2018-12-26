from fastai import *
from fastai.vision import *
from dist import env_world_size, env_rank, reduce_tensor
from loss import TransferLoss

# Callbacks
class DistributedRecorder(Recorder):
    def __init__(self, learn:Learner, save_path, print_freq=50):
        super().__init__(learn)
        self.save_path,self.print_freq = save_path,print_freq
    
    def on_train_begin(self, pbar:PBar, metrics_names:Collection[str], **kwargs:Any)->None:
        self.b_count = 0
        super().on_train_begin(pbar, metrics_names, **kwargs)
        
    def on_backward_begin(self, last_loss:Tensor, smooth_loss:Tensor, **kwargs:Any)->None:
        self.b_count += 1
        if env_world_size() > 1:
            metrics = smooth_loss.clone().detach().float().cuda()
            smooth_loss = reduce_tensor(metrics).cpu().numpy()

        if self.b_count % self.print_freq == 0:
            print(f'[{self.b_count}/{len(self.learn.data.train_dl)}]\tLosses:', smooth_loss)
            
        super().on_backward_begin(smooth_loss.sum())
        return last_loss.sum()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        if env_rank() == 0:
            name = Path(self.save_path).stem
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
