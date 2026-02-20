import torch
import random
import numpy as np

def init_seed(seed):
    # 启用 cuDNN 加速
    torch.backends.cudnn.enabled = True
    # 自动寻找最优算子执行路径（A100/A6000 必开）
    torch.backends.cudnn.benchmark = True 
    # 除非你正在进行非常严格的数学调试，否则不要设为 True
    torch.backends.cudnn.deterministic = False 

    # 开启 A100/A6000 的 TF32 模式
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_device(opt):
    if torch.cuda.is_available():
        opt.cuda = True
        torch.cuda.set_device(int(opt.device[5]))
    else:
        opt.cuda = False
        opt.device = 'cpu'
    return opt

def init_optim(model, opt):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),lr=opt.lr_init)

def init_lr_scheduler(optim, opt):
    '''
    Initialize the learning rate scheduler
    '''
    #return torch.optim.lr_scheduler.StepLR(optimizer=optim,gamma=opt.lr_scheduler_rate,step_size=opt.lr_scheduler_step)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=opt.lr_decay_steps,
                                                gamma = opt.lr_scheduler_rate)

def print_model_parameters(model, logger, only_num = True):
    logger.info('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            logger.info('{} {} {}'.format(name, param.shape, param.requires_grad))
    total_num = sum([param.nelement() for param in model.parameters()])
    logger.info('Total params num: {}'.format(total_num))
    logger.info('*****************Finish Parameter****************')

def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024*1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024*1024.)
    print('Allocated Memory: {:.2f} MB, Cached Memory: {:.2f} MB'.format(allocated_memory, cached_memory))
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    return allocated_memory, cached_memory

def init_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.uniform_(p)
    return model

def MAE_torch(output, label):
    return torch.mean(torch.abs(output - label))

def RMSE_torch(output, label):
    return torch.sqrt(torch.mean((output - label)**2))

def MAPE_torch(output, label):
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    label = torch.where(torch.abs(label) < 1e-4, torch.zeros_like(label), label)
    if np.isnan(null_val):
        mask = ~torch.isnan(label)
    else:
        eps = 5e-5
        mask = ~torch.isclose(label, torch.tensor(null_val).expand_as(label).to(label.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # loss = torch.abs(torch.abs(output-label)/label)
    # loss = loss * mask
    # loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # return torch.mean(loss)
    
    # mask = ~torch.isnan(label)
    # mask = mask.float()
    # mask /=  torch.mean((mask))
    # mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = 2.0 * (torch.abs(output - label) / (torch.abs(output) + torch.abs(label)))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
    # # return torch.mean(torch.abs(output - label)/label)

def NMAE_torch(output, label, capacity):
    n = capacity.shape[0]
    capacity = capacity.view(1, 1, n, 1)
    ratio = torch.where(capacity > 1e-4, torch.abs(output - label) / capacity, torch.nan)
    return torch.nanmean(ratio)

def All_Metrics(pred, true, capacity, mask1, mask2):
    mae  = MAE_torch(pred, true)
    rmse = RMSE_torch(pred, true)
    mape = MAPE_torch(pred, true)
    nmae = NMAE_torch(pred, true, capacity)
    rrse = 0.0
    corr = 0.0
    return mae, rmse, mape, nmae, rrse, corr

