import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import torch.nn.functional as F
import digital_diffeomorphism as dd


def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

def main():
    weights = [1, 1, 1] # loss weights
    model_folder = 'TransMorphTVF_ncc_{}_diffusion_{}_dsc_{}/'.format(weights[0], weights[1], weights[2])
    model_dir = 'experiments/' + model_folder
    results_folder = 'Quantitative_Results/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    csv_name = model_folder[:-1]
    csv_writter('Pat_ID,clus_0,clus_1,clus_2,clus_3,NDV,NPDJ,JacStd', results_folder + csv_name)

    #outdir = 'disps/'
    test_dir1 = 'G:/DATA/ACDC/test/'
    test_dir2 = 'G:/DATA/MM_processed/Reordered/test/'
    model_idx = -1
    '''
    Initialize model
    '''
    H, W, D = 128, 128, 32
    config = CONFIGS_TM['TransMorph-3-LVL']
    config.img_size = (H, W, D)
    config.window_size = (H // 32, W // 32, D // 32)
    config.out_chan = 3
    model = TransMorph.TransMorphTVF(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    spatial_trans = TransMorph.SpatialTransformer((H, W, D)).cuda()
    print(spatial_trans.grid)
    '''
    Initialize training
    '''
    test_set = datasets.ACDCInferDataset(glob.glob(test_dir1+'*pkl')+glob.glob(test_dir2+'*pkl'))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    idx = 0
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        for data in test_loader:
            model.eval()
            x = data[0].cuda().float()
            y = data[1].cuda().float()
            x_seg = data[2].cuda()
            y_seg = data[3].cuda()
            x_seg_oh = F.one_hot(x_seg.long().cuda(), 4).float().squeeze(1).permute(0, 4, 1, 2, 3).cuda()
            flow = model((x, y))

            def_seg = spatial_trans(x_seg_oh, flow)
            def_out = torch.argmax(def_seg, dim=1, keepdim=True)

            y_mask = torch.sum(torch.sum(y, dim=2, keepdim=True), dim=3, keepdim=True)
            y_mask = (y_mask > 0).float()

            mask = x.cpu().detach().numpy()[0, 0, 1:-1, 1:-1, 1:-1]
            mask = mask > 0
            disp_field = flow.cpu().detach().numpy()[0]
            trans_ = disp_field + dd.get_identity_grid(disp_field)
            jac_dets = dd.calc_jac_dets(trans_)
            non_diff_voxels, non_diff_tetrahedra, non_diff_volume, non_diff_volume_map = dd.calc_measurements(jac_dets,
                                                                                                              mask)
            total_voxels = np.sum(mask)
            jac_det_vol = non_diff_volume / total_voxels * 100
            jac_det_all = np.sum((jac_dets['all J_i>0'] <= 0)) / np.prod(mask.shape) * 100
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), idx)

            disp_field = flow.cpu().detach().numpy()[0]
            jac_det_ = (utils.jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001,
                                                                                                 1000000000)
            log_jac_det = np.log(jac_det_).std()

            line = line + ',' + str(jac_det_vol) + ',' + str(jac_det_all) + ',' + str(
                log_jac_det)
            csv_writter(line, results_folder + csv_name)
            eval_det.update(jac_det_vol, x.size(0))
            print('None diff vol: {}'.format(jac_det_vol))
            dsc_trans = utils.dice_val((def_out * y_mask).long(), y_seg.long(), 4)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 4)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))

            idx += 1

        print('Deformed DSC: {:.4f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[0], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_img.shape[1], grid_step):
        grid_img[:, i+line_thickness-1, :] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    torch.manual_seed(0)
    main()