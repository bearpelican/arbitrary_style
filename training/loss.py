from torch import nn
import torch
from fastai import *

class TransferLoss(nn.Module):
    def __init__(self, m_vgg, cont_wgt, style_wgt, style_block_wgts, tva_wgt, data_norm, c_block=1):
        super().__init__()
        self.style_wgt,self.style_block_wgts = style_wgt,style_block_wgts
        self.cont_wgt,self.tva_wgt = cont_wgt,tva_wgt
        self.m_vgg,self.data_norm,self.c_block = m_vgg,data_norm,c_block
        
    def forward(self, input, x_cont, x_style):
        style_wgts = [self.style_wgt*b for b in self.style_block_wgts]
        bs = x_cont.size(0)
        with torch.no_grad(): 
            style_batch = x_style.repeat(bs,1,1,1)
            s_out = self.m_vgg(style_batch)
            style_feat = [s.clone() for s in s_out]
            
            cont_feat = self.m_vgg(x_cont)[self.c_block].clone()
            
        input_norm,_ = self.data_norm((input,None))
        inp_feat = self.m_vgg(input_norm)
        
        closs = [ct_loss(inp_feat[self.c_block],cont_feat) * self.cont_wgt]
        sloss = [gram_loss(inp,targ)*wgt for inp,targ,wgt in zip(inp_feat, style_feat, style_wgts) if wgt > 0]
        tvaloss = tva_loss(input_norm) * self.tva_wgt
        
        return torch.stack((sum(closs), sum(sloss), tvaloss))
        
# Loss Functions
# losses
def ct_loss(input, target): return F.mse_loss(input,target)

def gram(input):
        b,c,h,w = input.size()
        x = input.view(b, c, -1)
        return torch.bmm(x, x.transpose(1,2))/(c*h*w)

def gram_loss(input, target): return F.mse_loss(gram(input), gram(target))

def tva_loss(y):
    w_var = torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:]))
    h_var = torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
    return w_var + h_var
