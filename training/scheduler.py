from fastai import *

# ### Learning rate scheduler
class Scheduler():
    def __init__(self, phases, key):
        self.current = None
        self.key = key
        self.phases = [self.format_phase(p) for p in phases]
        self.tot_epochs = max([max(p['ep']) for p in self.phases])

    def format_phase(self, phase):
        phase['ep'] = listify(phase['ep'])
        phase[self.key] = listify(phase[self.key])
        if len(phase[self.key]) == 2: 
            assert (len(phase['ep']) == 2), 'Linear learning rates must contain end epoch'
        return phase

    def linear_phase(self, phase, epoch, batch_curr, batch_tot):
        x_start, x_end = phase[self.key]
        ep_start, ep_end = phase['ep']
        if epoch > ep_end: return x_end
        ep_relative = epoch - ep_start
        ep_tot = ep_end - ep_start
        return self.calc_linear(x_start, x_end, ep_relative, batch_curr, ep_tot, batch_tot)

    def calc_linear(self, x_start, x_end, epoch_curr, batch_curr, epoch_tot, batch_tot):
        step_tot = epoch_tot * batch_tot
        step_curr = epoch_curr * batch_tot + batch_curr 
        step_size = (x_end - x_start)/step_tot
        return x_start + step_curr * step_size
    
    def get_current_phase(self, epoch):
        for phase in reversed(self.phases): 
            if (epoch >= phase['ep'][0]): return phase
        raise Exception('Epoch out of range')
            
    def get_val(self, epoch, batch_curr, batch_tot):
        phase = self.get_current_phase(epoch)
        if len(phase[self.key]) == 1: return phase[self.key][0] # constant learning rate
        return self.linear_phase(phase, epoch, batch_curr, batch_tot)

# ### Learning rate scheduler
class LRScheduler(Scheduler):
    def __init__(self, optimizer, phases):
        super().__init__(phases, 'lr')
        self.optimizer = optimizer

    def update_lr(self, epoch, batch_num, batch_tot):
        lr = self.get_val(epoch, batch_num, batch_tot) 
        if self.current == lr: return
        if ((batch_num == 1) or (batch_num == batch_tot)): 
            print(f'Changing LR from {self.current} to {lr}')

        self.current = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
