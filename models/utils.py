import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def adjust_learning_rate(optimizer, epoch, decay_rate, decay_every, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init * (decay_rate ** (epoch // decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr