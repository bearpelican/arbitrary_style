from fastai import *
from torchvision.models import vgg16_bn, resnet50, inception_v3


# Combined Transformer+Predictor
class CombinedModel(nn.Module):
    def __init__(self, m_tran, m_style):
        super().__init__()
        self.m_tran, self.m_style = m_tran, m_style
        
    def forward(self, x_con, x_style):
        s_out = self.m_style(x_style)
        return self.m_tran(x_con, s_out)

# Style Transformer

class StyleInstanceNorm2d(nn.InstanceNorm2d):
    def forward(self, input):
        self._check_input_dim(input)
        weight = self.style_weight if hasattr(self, 'style_weight') else self.weight
        bias = self.style_bias if hasattr(self, 'style_bias') else self.bias
        
            
        return F.instance_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)
    
    def reset_style(self):
        self.style_weight = None
        self.style_bias = None

def get_inorm(m):
    ls = []
    if isinstance(m, StyleInstanceNorm2d): ls.append(m)
    for mod in m.children():
        ls.extend(get_inorm(mod))
    return ls # skip downsample instance norm

def conv(ni, nf, kernel_size=3, stride=1, actn=True, pad=None, inorm=StyleInstanceNorm2d, bnorm=False):
    if pad is None: pad = kernel_size//2
    layers = [nn.ReflectionPad2d(pad), nn.Conv2d(ni, nf, kernel_size, stride=stride, bias=(not bnorm and not inorm))]
    if bnorm: layers.append(nn.BatchNorm2d(nf))
    if inorm: layers.append(inorm(nf, affine=True))
    if actn: layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def res_block(nf):
    return nn.Sequential(conv(nf, nf, actn=True, pad=None), conv(nf, nf, actn=False, pad=None))

def upsample(ni, nf):
    return nn.Sequential(nn.Upsample(scale_factor=2), conv(ni, nf))

class StyleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        features = [conv(3, 32, 9, inorm=nn.InstanceNorm2d),
                    conv(32, 64, stride=2, inorm=nn.InstanceNorm2d), 
                    conv(64, 128, stride=2, inorm=nn.InstanceNorm2d)]
        for i in range(5): features.append(res_block(128))
        features += [upsample(128, 64), 
                     upsample(64, 32),
                     conv(32, 3, 9, actn=False), nn.Sigmoid()]
        self.features = nn.Sequential(*features)
        self.inorms = get_inorm(self.features)[-4:]
        
    def forward(self, x, style_pred): 
        if style_pred is not None: 
            self.update_inorm(style_pred.squeeze())
        else:
            print('Resetting style')
            for m in self.inorms: m.reset_style()
                
        return self.features(x)
    
    def update_inorm(self, iparams):
        index = 0
        for m_in in reversed(self.inorms):
            m_in.style_weight = iparams[index:(index+m_in.weight.size(0))]
            index = index + m_in.weight.size(0)
            m_in.style_bias = iparams[index:(index+m_in.bias.size(0))]
            index = index + m_in.bias.size(0)
            if index > len(iparams): print('Ran out of parameters!!', index, len(iparams))


# Style Predictor

class StylePredict(nn.Module):
    def __init__(self, pretrained, cut_name='avgpool', in_features=2048, out_features=2758, conv=False):
        super().__init__()
        cut = [i for i,(name,mod) in enumerate(pretrained.named_children()) if name == cut_name]
        self.base = nn.Sequential(*list(pretrained.children())[:(cut[0])])
        if conv:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                      nn.Conv2d(in_features, 100, 1, stride=1), #nn.ReLU(inplace=True), 
                                      nn.Conv2d(100, out_features, 1, stride=1))
        else:
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten(),
                                      nn.Linear(in_features, 100), nn.ReLU(inplace=True), 
                                      nn.Linear(100, out_features))
        for p in self.base.parameters():
            p.requires_grad_(False)
            
    def forward(self, x):
        return self.head(self.base(x))


    @classmethod
    def create_inception(cls):
        # inception_v3 is missing a max pool - MaxPool_3a_3x3
        m_arch = inception_v3(pretrained=True)
        return StylePredict(m_arch, cut_name='AuxLogits', in_features=768, conv=True)

    @classmethod
    def create_resnet(cls):
        m_arch = resnet50(pretrained=True)
        return StylePredict(m_arch, cut_name='layer4', in_features=1024, conv=True)


# VGG

class VGGActivations(nn.Module):
    def __init__(self):
        super().__init__()
        m_vgg = vgg16_bn(True).eval()
        self.base = children(m_vgg)[0]
        requires_grad(self.base, False)

        blocks = [i-1 for i,o in enumerate(children(self.base))
                    if isinstance(o,nn.MaxPool2d)]

        # Note: starts on block 1
        layer_ids = blocks[1:]
        print('Layer ids: ', layer_ids)
        self.hooks = Hooks([self.base[i] for i in layer_ids], lambda m,i,o: o, detach=False)

    def forward(self, input):
        self.base(input)
        return self.hooks.stored
