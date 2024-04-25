from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys, random
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph_Bayes import CONFIGS as CONFIGS_TM
import models.TransMorph_Bayes as TransMorph
import torch.nn.functional as F

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    batch_size = 1
    weights = [1, 1, 1] # loss weights
    train_dir1 = 'G:/DATA/ACDC/train/'
    train_dir2 = 'G:/DATA/MM_processed/Reordered/train/'
    val_dir1 = 'G:/DATA/ACDC/val/'
    val_dir2 = 'G:/DATA/MM_processed/Reordered/val/'
    save_dir = 'TransMorphTVFUncertBayes_ncc_{}_diffusion_{}_nll_{}_dsc_1/'.format(weights[0], weights[1], weights[2])
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)

    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 #learning rate
    epoch_start = 0
    max_epoch = 500 #max traning epoch
    cont_training = False #if continue training
    '''
    Initialize model
    '''
    H, W, D = 128, 128, 32
    config = CONFIGS_TM['TransMorph-3-LVL-Bayes']
    config.img_size = (H, W, D)
    config.window_size = (H // 32, W // 32, D // 32)
    config.out_chan = 3
    model = TransMorph.TransMorphTVFUncert(config)
    uncert = TransMorph.UncertLayer(4, config.embed_dim)
    model.cuda()
    uncert.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model((H, W, D), 'nearest')
    reg_model.cuda()
    spatial_trans = TransMorph.SpatialTransformer((H, W, D)).cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 201
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_set = datasets.ACDCMMDataset(glob.glob(train_dir1+'*pkl')+glob.glob(train_dir2+'*pkl'))
    val_set = datasets.ACDCMMDataset(glob.glob(val_dir1+'*pkl')+glob.glob(val_dir2+'*pkl'))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    params = [{'params': model.parameters(), 'lr': updated_lr}] + [{'params': uncert.parameters(), 'lr': updated_lr}]
    optimizer = optim.AdamW(params, lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion_ncc = losses.NCC_vxm()
    criterion_reg = losses.Grad3d(penalty='l2')
    criterion_nll = losses.BetaNLLLoss()
    criterion_dsc = losses.DiceLoss()
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            uncert.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            with torch.no_grad():
                x_ = data[0].cuda().float()
                y_ = data[1].cuda().float()
                x_seg_ = data[2].cuda().float()
                y_seg_ = data[3].cuda().float()

            x, y, x_seg, y_seg = utils.flip_aug(x_, y_, x_seg_, y_seg_)
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=4)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()

            flow, xx = model((x,y))
            #flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear') * 2
            output = model.spatial_trans(x, flow)
            err = (output-y)**2
            sigma2 = uncert(err, xx)
            def_seg = model.spatial_trans(x_seg_oh.float(), flow.float())
            loss_ncc = criterion_ncc(output, y) * weights[0]
            loss_reg = criterion_reg(flow, y) * weights[1]
            loss_nll = criterion_nll(def_seg, y_seg.long(), sigma2) * weights[2]
            loss_dsc = criterion_dsc(def_seg, y_seg.long())
            loss = loss_ncc + loss_reg + loss_dsc + loss_nll
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            x, y, x_seg, y_seg = utils.flip_aug(x_, y_, x_seg_, y_seg_)
            y_seg_oh = nn.functional.one_hot(y_seg.long(), num_classes=4)
            y_seg_oh = torch.squeeze(y_seg_oh, 1)
            y_seg_oh = y_seg_oh.permute(0, 4, 1, 2, 3).contiguous()

            flow, xx = model((y, x))
            #flow = F.interpolate(flow.cuda(), scale_factor=2, mode='trilinear') * 2
            output = model.spatial_trans(y, flow)
            err = (output - x) ** 2
            sigma2 = uncert(err, xx)
            def_seg = model.spatial_trans(y_seg_oh.float(), flow.float())
            loss_ncc = criterion_ncc(output, x) * weights[0]
            loss_reg = criterion_reg(flow, x) * weights[1]
            loss_nll = criterion_nll(def_seg, x_seg.long(), sigma2) * weights[2]
            loss_dsc = criterion_dsc(def_seg, x_seg.long())
            loss = loss_ncc + loss_reg + loss_dsc + loss_nll
            loss_all.update(loss.item(), x.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Seg NLL: {:.6f}, Seg Dice: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader),
                                                                                            loss.item(),
                                                                                            loss_ncc.item(),
                                                                                            loss_nll.item(),
                                                                                            loss_dsc.item(),
                                                                                            loss_reg.item()))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        dsc_raw = []
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                uncert.eval()
                x = data[0].cuda().float()
                y = data[1].cuda().float()
                x_seg = data[2]
                y_seg = data[3]
                y_mask = torch.sum(torch.sum(y, dim=2, keepdim=True), dim=3, keepdim=True)
                y_mask = (y_mask > 0).float()
                grid_img = mk_grid_img(8, 1, grid_sz=(H, W, D))

                x_def_segs, x_def_seg_vars, flows, errs = utils.get_mc_seg_preds_w_errors(model, uncert, (x.cuda(), y.cuda()), (x_seg.cuda(), y_seg.cuda()), 5)
                min_err_idx = np.argmin(errs)
                flow_min = flows[min_err_idx]
                uncert_epi = torch.var(torch.cat(x_def_segs, dim=0)[:], dim=0, keepdim=True)
                uncert_alc = torch.mean(torch.cat(x_def_seg_vars, dim=0)[:], dim=0, keepdim=True)

                x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=4)
                x_seg_oh = torch.squeeze(x_seg_oh, 1)
                x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
                def_seg = spatial_trans(x_seg_oh.cuda().float(), flow_min.cuda())
                def_seg = torch.argmax(def_seg, dim=1, keepdim=True)
                def_out = spatial_trans(x.cuda().float(), flow_min.cuda())
                def_grid = spatial_trans(grid_img.float(), flow_min.cuda())
                dsc = utils.dice_val_VOI((def_seg*y_mask).long(), y_seg.long())
                if epoch == 0:
                    dsc_raw.append(utils.dice_val_VOI(x_seg.long(), y_seg.long()).item())
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        if epoch == 0:
            print('raw dice: {}'.format(np.mean(dsc_raw)))
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_model': model.state_dict(),
            'state_dict_uncert': uncert.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/'+save_dir, filename='dsc{:.4f}.pth.tar'.format(eval_dsc.avg))
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        uncertepi_fig = comput_fig(uncert_epi)
        uncertalc_fig = comput_fig(uncert_alc)
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x)
        tar_fig = comput_fig(y)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        writer.add_figure('uncertainty epi', uncertepi_fig, epoch)
        plt.close(uncertepi_fig)
        writer.add_figure('uncertainty alc', uncertalc_fig, epoch)
        plt.close(uncertalc_fig)
        loss_all.reset()
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, :, :, 10:26]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[2]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[:, :, i], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_img.shape[1], grid_step):
        grid_img[:, i+line_thickness-1, :] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def seedBasic(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seedTorch(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    DEFAULT_RANDOM_SEED = 42

    seedBasic(DEFAULT_RANDOM_SEED)
    seedTorch(DEFAULT_RANDOM_SEED)
    main()