# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
import math
# Numerical libs
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from data.augmentations import Compose, Scale, RandomSizedCrop, AdjustContrast, AdjustBrightness, RandomVerticallyFlip, RandomHorizontallyFlip, RandomRotate, PaddingCenterCrop
# Our libs
from models.models_unet import ModelBuilder, SegmentationModule
from data.hcm_AllData_cv import Dataloader2D as newTorontoData
from utils import AverageMeter, parse_devices, accuracy, intersectionAndUnion, Dice
from lib.nn import UserScatteredDataParallel, async_copy_to,  user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata
from lib.utils import as_numpy
import numpy as np
from loss_unet import DualLoss
from radam import RAdam
import sklearn.metrics
import cv2
from PIL import Image as im

#import resnet3d

def visualize_result_attention(data, seg, pred, att, idx, args, p_id):
    img = data[0][0]
    img = img.detach().cpu().numpy()
    #normalize image to [0, 1] first.
    img = (img - img.min())/(img.max()-img.min())
    img = (img * 255).astype(np.uint8) #Then scale it up to [0, 255] to get the final image.
    pred_img = (pred * 85).astype(np.uint8)
    seg = (seg[0].detach().cpu().numpy()*85).astype(np.uint8)
    seg = np.squeeze(seg, 0) 

    att2 = att[0][0][0].detach().cpu().numpy()*255
    att3 = att[1][0][0].detach().cpu().numpy()*255
    att4 = att[2][0][0].detach().cpu().numpy()*255
    att5 = att[3][0][0].detach().cpu().numpy()*255
    g1 = att[4][0][0]*255
    g2 = att[5][0][0]*255
    g3 = att[6][0][0]*255
    edge_out = att[7][0].detach().cpu().numpy()*255

    #print(img.shape, pred_img.shape)
    #heat = get_heatmap(LRP)
    im_vis1 = np.concatenate((img, seg, pred_img, att2, att3), axis=1).astype(np.uint8)
    im_vis2 = np.concatenate((att4, att5, g1, g2, g3), axis=1).astype(np.uint8)
    im_vis = np.concatenate((im_vis1, im_vis2), axis=0).astype(np.uint8)
    img_name = str(p_id) + "_" + str(idx) + '.png'
    
    cv2.imwrite(os.path.join('/cluster/projects/bwanggroup/HCM_MRI/shared/code/attention_210222',
                img_name), im_vis)

def visualize_result(img, label, pred, idx, args, pat):
    #normalize image to [0, 1] first.
    img = img[0][0].unsqueeze(0)
    img = (img - img.min())/(img.max()-img.min())
    img = (img.cpu().numpy() * 255).astype(np.uint8) #Then scale it up to [0, 255] to get the final image.
    label[0] = (label[0].cpu().numpy() * 85).astype(np.uint8)
    label[1] = (label[1].cpu().numpy() * 85).astype(np.uint8)
    pred_img = (pred * 85).astype(np.uint8)
    #print(type(img))
    #print(img.shape)
    #print(label[0].shape)
    #print(label[1].shape)
    #print(type(pred_img))
    #print(np.expand_dims(pred_img, axis=0).shape)
    # im_vis = np.concatenate((img, label, pred_img), axis=1).astype(np.uint8)
    print(img.shape, label[0].shape, pred_img.shape)
    im_vis = np.concatenate((img, label[0], np.expand_dims(pred_img, axis=0)), axis=1).astype(np.uint8)
    img_name = pat+ '_' + str(idx) + '.png'
    cv2.imwrite(os.path.join('/cluster/projects/bwanggroup/HCM_MRI/shared/code/crop_zoom_unet/',
                img_name), im_vis.transpose(1, 2, 0))

'''
def visualize_result(img, label, pred, idx, args, name):
    img_name = str(name) + '.png'
    #normalize image to [0, 1] first.
    s = 256 #8192
    img = img[0][0]
    img = (img - img.min())/(img.max()-img.min())
    img = (img.cpu().numpy() * 255).astype(np.uint8) #Then scale it up to [0, 255] to get the final image.
    img = im.fromarray(img)
    img = cv2.resize(np.float32(img), (s,s), interpolation=cv2.INTER_LINEAR)
    img_tmp = np.expand_dims(np.asarray(img), axis=0)
    tmp_name = 'img_'+img_name
    tmp = cv2.imwrite(os.path.join('/cluster/projects/bwanggroup/HCM_MRI/shared/code/imgs/visualized_imgs79/',
                tmp_name), np.squeeze(img_tmp, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 100])
    print('img:')
    print(type(np.squeeze(img_tmp, axis=0)))
    print(np.squeeze(img_tmp, axis=0).shape)
    print(tmp)

    label = (np.asarray(label[0]) * 85).astype(np.uint8)
    label = im.fromarray(np.squeeze(label, axis=0))
    label = cv2.resize(np.float32(label), (s,s), interpolation=cv2.INTER_NEAREST)
    label_tmp = np.expand_dims(np.asarray(label), axis=0)
    tmp_name = 'label_'+img_name
    tmp = cv2.imwrite(os.path.join('/cluster/projects/bwanggroup/HCM_MRI/shared/code/imgs/visualized_imgs79/',
                tmp_name), np.squeeze(label_tmp, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 100])
    print('label:')
    print(type(np.squeeze(label_tmp, axis=0)))
    print(np.squeeze(label_tmp, axis=0).shape)
    print(tmp)

    pred_img = (pred * 85).astype(np.uint8)
    pred_img = im.fromarray(pred_img)
    pred_img = cv2.resize(np.float32(pred_img), (s,s), interpolation=cv2.INTER_NEAREST)

    im_vis = np.concatenate((np.expand_dims(np.asarray(img), axis=0), np.expand_dims(np.asarray(label), axis=0), np.expand_dims(np.asarray(pred_img), axis=0)), axis=1).astype(np.uint8)
    # print('im_vis')
    # print(type(im_vis))
    # print(im_vis.shape)
    tmp = cv2.imwrite(os.path.join('/cluster/projects/bwanggroup/HCM_MRI/shared/code/imgs/visualized_imgs79/',
                img_name), np.squeeze(im_vis, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 100])
    

    print('np.squeeze(im_vis, axis=0)')
    print(type(np.squeeze(im_vis, axis=0)))
    print(np.squeeze(im_vis, axis=0).shape)
    print(tmp)
'''

def get_concat_h(im1, im2):
    dst = im.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

'''
def visualize_result(img, label, pred, idx, args, name):
    address = '/cluster/projects/bwanggroup/HCM_MRI/shared/code/imgs/visualized_imgs86/'
    img_name = str(name) + '.png'
    #normalize image to [0, 1] first.
    s = 1024
    img = img[0][0]
    img = (img - img.min())/(img.max()-img.min())
    img = (img.cpu().numpy() * 255).astype(np.uint8) #Then scale it up to [0, 255] to get the final image.
    img = im.fromarray(img)
    img = cv2.resize(np.float32(img), (s,s), interpolation=cv2.INTER_LINEAR)
    img_tmp = np.expand_dims(np.asarray(img), axis=0)
    tmp_name = 'img_'+img_name
    img_archive = np.squeeze(img_tmp, axis=0)
    tmp = cv2.imwrite(os.path.join(address, tmp_name), np.squeeze(img_tmp, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 100])
    # print(tmp)
                
    label = (np.asarray(label[0]) * 85).astype(np.uint8)
    label = im.fromarray(np.squeeze(label, axis=0))
    label = cv2.resize(np.float32(label), (s,s), interpolation=cv2.INTER_NEAREST)
    label_tmp = np.expand_dims(np.asarray(label), axis=0)
    tmp_name = 'label_'+img_name
    tmp = cv2.imwrite(os.path.join(address, tmp_name), np.squeeze(label_tmp, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 100])
    # print(tmp)

    pred_img = (pred * 85).astype(np.uint8)
    pred_img = im.fromarray(pred_img)
    pred_img = cv2.resize(np.float32(pred_img), (s,s), interpolation=cv2.INTER_NEAREST)
    pred_img_tmp = np.expand_dims(np.asarray(pred_img), axis=0)
    tmp_name = 'pred_'+img_name
    tmp = cv2.imwrite(os.path.join(address, tmp_name), np.squeeze(pred_img_tmp, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 100])
    # print(tmp)


    # read label
    ## approach 1
    # https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
    tmp_name = 'label_'+img_name
    image = cv2.imread(os.path.join(address, tmp_name))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # image = cv2.imread(os.path.join(address, 'img_'+img_name))
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    mask = np.ones(image.shape, dtype=np.uint8) * 255
    cv2.drawContours(mask, contours[0], -1, (0,255,0), thickness = 2)

    tmp_name = 'edged_label_'+img_name
    print('approach 1 input label:')
    print(type(image))
    print(image.shape)
    # tmp = cv2.imwrite(os.path.join(address, tmp_name), np.squeeze(edged_tmp, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 100])
    tmp = cv2.imwrite(os.path.join(address, tmp_name), mask, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(tmp)

    
    tmp_name = 'pred_'+img_name
    image = cv2.imread(os.path.join(address, tmp_name))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    image = cv2.imread(os.path.join(address, 'img_'+img_name))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    mask = np.ones(image.shape, dtype=np.uint8) * 255
    cv2.drawContours(mask, contours[0], -1, (0,255,0), thickness = 2)

    tmp_name = 'edged_pred_'+img_name
    print('approach 1 input pred:')
    print(type(image))
    print(image.shape)
    # tmp = cv2.imwrite(os.path.join(address, tmp_name), np.squeeze(edged_tmp, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 100])
    tmp = cv2.imwrite(os.path.join(address, tmp_name), mask, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(tmp)
'''

def eval_r2(loader_val, segmentation_module, args, crit):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    loss_meter = AverageMeter()
    scar_i = AverageMeter()
    scar_u = AverageMeter()
    scarExists = False
    pred_area = []
    true_area = []

    segmentation_module.eval()
    for idx, batch_data in enumerate(loader_val):
        seg_label = as_numpy(batch_data[1][0])
        #print('seg_label:')
        #print(seg_label.shape)
        #print(np.amax(seg_label))
        #print(np.amin(seg_label))
        torch.cuda.synchronize()
        batch_data[0] = batch_data[0][0].unsqueeze(0).cuda()

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, args.gpu)
            feed_dict = batch_data.copy()
            #print(torch.max(feed_dict['image']))

            # forward pass
            scores_tmp, loss = segmentation_module(feed_dict, epoch=0, segSize=segSize)
            scores = scores + scores_tmp
            loss_meter.update(loss)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
            scar_area = np.sum((pred==3).astype(np.uint32))
            gt_area = np.sum((seg_label==3).astype(np.uint32))
            pred_area.append(scar_area)
            true_area.append(gt_area)
        #visualize_result(batch_data[0], batch_data[1], pred, idx, args, batch_data[5])

        torch.cuda.synchronize()
        # calculate accuracy
        intersection, union = intersectionAndUnion(pred, seg_label, args.num_class)
        #print(np.amax(seg_label))
        if np.amax(seg_label) == 3:
            scarExists=True
            scar_i.update(intersection[-1])
            scar_u.update(union[-1])

        intersection_meter.update(intersection[:-1])
        union_meter.update(union[:-1])

    r2 = sklearn.metrics.r2_score(true_area, pred_area)
    print("R2 SCORE: " + str(r2))
    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    if scarExists==True:
        iou = np.append(iou, scar_i.sum / (scar_u.sum + 1e-10))

    for i, _iou in enumerate(iou):
        if i >= 1:
            print('class [{}], IoU: {:.4f}'.format(i, _iou))
    print('loss: {:.4f}'.format(loss_meter.average()))
    return iou[1:], loss_meter.average()

def eval(loader_val, segmentation_module, args, crit):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    loss_meter = AverageMeter()
    scarExists = False
    scar_i = AverageMeter()
    scar_u = AverageMeter()
    t = AverageMeter()
    iou_meter = AverageMeter()

    iou_list_epi = np.array([])
    iou_list_endo = np.array([])
    iou_list_scar = np.array([])
    
    intersection_file = open('iou_intersection_allData_baseline-saunet-ngpus1-batchSize8-LR_unet0.0005-epoch300_unet_epoch_237.txt', "w+")
    union_file = open('iou_union_allData_baseline-saunet-ngpus1-batchSize8-LR_unet0.0005-epoch300_unet_epoch_237.txt', "w+")
    
    cur_patient = None
    slice_index = 0
    patient_i = AverageMeter()
    patient_u = AverageMeter()

    segmentation_module.eval()
    for idx, batch_data in enumerate(loader_val):
        seg_label = as_numpy(batch_data[1][0])
        torch.cuda.synchronize()
        batch_data[0] = batch_data[0][0].unsqueeze(0).cuda()

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, args.gpu)
            feed_dict = batch_data.copy()

            # forward pass
            tic = time.time()
            scores_tmp, loss = segmentation_module(feed_dict, epoch=0, segSize=segSize)
            scores = scores + scores_tmp
            loss_meter.update(loss)
            t.update(time.time() - tic)
            tic = time.time()

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
            if batch_data[-1][0] != cur_patient:
                '''
                if cur_patient != None and slice_index > 0:
                    #print(patient_i.sum, patient_u.sum)
                    iou = patient_i.sum / (patient_u.sum+1e-10)
                    if scarExists:
                        iou_list_scar = np.append(iou_list_scar, 2*iou[-1]/(iou[-1]+1))
                    iou_list_epi = np.append(iou_list_epi, 2*iou[1]/(iou[1]+1))
                    iou_list_endo = np.append(iou_list_endo, 2*iou[2]/(iou[2]+1))
                    patient_i = AverageMeter()
                    patient_u = AverageMeter()
                    scarExists=False
                '''
                slice_index = 0
                cur_patient = batch_data[-1][0]

            else:
                slice_index += 1

        #visualize_result_attention(batch_data[0], batch_data[1], pred, att, slice_index, args, cur_patient)
        #visualize_result(batch_data[0], batch_data[1], pred, idx, args, batch_data[-1][0])
        
        torch.cuda.synchronize()
        # calculate accuracy
        ''' commented this out, may 28, 2021'''
        intersection, union = intersectionAndUnion(pred, seg_label, args.num_class)
        '''
        patient_i.update(intersection)
        patient_u.update(union)
        '''
        #dice = Dice(pred, seg_label, args.num_class)
        #dice = newDiceCoefEvalBoston(seg_label, pred, args.num_class)
        #scar_iou = 0
        if np.amax(seg_label) == 3 or np.amax(pred) == 3:
            scarExists=True
            scar_i.update(intersection[-1])
            scar_u.update(union[-1])
            #iou_list_scar = np.append(iou_list_scar, intersection[-1]/(union[-1]+ 1e-10))
        '''
	else:
            intersection[-1] = -1 # it was 3
            union[-1] = -1 # it was 3
	'''
        intersection_meter.update(intersection[:-1])
        union_meter.update(union[:-1])
        
        #iou_list_epi = np.append(iou_list_epi, dice[1])#intersection[1]/(union[1]+ 1e-10))
        #iou_list_endo = np.append(iou_list_endo, dice[2])#intersection[2]/(union[2]+ 1e-10))
        
        #print('shape of intersection_meter in the loop: '+str(intersection_meter.val.shape))

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    #print('shape of intersection_meter after loop: '+str(intersection_meter.val.shape))
    ##print('shape of epi, endo and scar: '+str(iou_list_epi.shape)+' '+str(iou_list_endo.shape)+' '+str(iou_list_scar.shape))
    #iou = np.array([np.mean(iou_list_epi), np.mean(iou_list_endo), np.mean(iou_list_scar)])
    ##print('iou shape: '+str(iou.shape))
   
    if scarExists==True:
        #tmp = np.mean(scar_i.val/(scar_u.val+1e-10))
        #iou = np.append(iou, tmp)
        iou = np.append(iou, scar_i.sum / (scar_u.sum + 1e-10))
    ##print('iou shape after adding scar iou: '+str(iou.shape))
    for i, _iou in enumerate(iou):
        if i >= 0:
            print('class [{}], Dice: {:.4f}'.format(i, _iou))
    print('loss: {:.4f}'.format(loss_meter.average()))
    print('avg time: {:.4f}'.format(t.average()))

    return iou[0:], loss_meter.average()

# train one epoch
def train(segmentation_module, loader_train, optimizers, history, epoch, args):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_j1 = AverageMeter()
    ave_j2 = AverageMeter()
    ave_j3 = AverageMeter()
    #ave_j4 = AverageMeter()

    segmentation_module.train(not args.fix_bn)

    # main loop
    tic = time.time()
    iter_count = 0

    if epoch == args.start_epoch and args.start_epoch > 1:
        scale_running_lr = ((1. - float(epoch-1) / (args.num_epoch)) ** args.lr_pow)
        args.running_lr_encoder = args.lr_encoder * scale_running_lr
        for param_group in optimizers[0].param_groups:
            param_group['lr'] = args.running_lr_encoder
    for batch_data in loader_train:
        data_time.update(time.time() - tic)
        batch_data[0] = batch_data[0].cuda()
        segmentation_module.zero_grad()

        #seg_label = as_numpy(batch_data[1][0])

        # forward pass
        loss, acc = segmentation_module(batch_data, epoch)
        loss = loss.mean()

        jaccard = acc[1]
        for j in jaccard:
            j = j.float().mean()
        acc = acc[0].float().mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        iter_count += args.batch_size_per_gpu

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        ave_j1.update(jaccard[0].data.item()*100)
        ave_j2.update(jaccard[1].data.item()*100)
        ave_j3.update(jaccard[2].data.item()*100)
        #ave_j4.update(jaccard[3].data.item()*100)

        if iter_count % (args.batch_size_per_gpu*10) == 0:
            # calculate accuracy, and display
            if args.unet==False:
                print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                        'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                        'Accuracy: {:4.2f}, Loss: {:.6f}'
                        .format(epoch, i, args.epoch_iters,
                        batch_time.average(), data_time.average(),
                        args.running_lr_encoder, args.running_lr_decoder,
                        ave_acc.average(), ave_total_loss.average()))
            else:
                print('Epoch: [{}/{}], Iter: [{}], Time: {:.2f}, Data: {:.2f},'
                        ' lr_unet: {:.6f}, Accuracy: {:4.2f}, Jaccard: [{:4.2f}, {:4.2f}, {:4.2f}], ' #, {:4.2f}], '
                        'Loss: {:.6f}'
                        .format(epoch, args.max_iters, iter_count,
                            batch_time.average(), data_time.average(),
                            args.running_lr_encoder, ave_acc.average(),
                            ave_j1.average(), ave_j2.average(), ave_j3.average(), #ave_j4.average(),
                            ave_total_loss.average()))

    #Average jaccard across classes.
    j_avg = (ave_j1.average()+ave_j2.average()+ave_j3.average())/3 #(ave_j1.average() + ave_j2.average() + ave_j3.average())/3

    #Update the training history
    history['train']['epoch'].append(epoch)
    history['train']['loss'].append(loss.data.item())
    history['train']['acc'].append(acc.data.item())
    history['train']['jaccard'].append(j_avg)
    # adjust learning rate
    adjust_learning_rate(optimizers, epoch, args)


def checkpoint(nets, history, args, epoch_num):
    print('Saving checkpoints...')
    (unet, crit) = nets
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))

    dict_unet = unet.state_dict()
    torch.save(dict_unet,
                '{}/unet_{}'.format(args.ckpt, suffix_latest))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, args):
    (unet, crit) = nets
    if args.optimizer.lower() == 'sgd':
        print("SGD Initialized")
        optimizer_unet = torch.optim.SGD(
            group_weight(unet),
            lr=args.lr_encoder,
            momentum=args.beta1,
            weight_decay=args.weight_decay,
            nesterov=False)
    elif args.optimizer.lower() == 'adam':
        print("Adam Initialized")
        optimizer_unet = torch.optim.Adam(
            group_weight(unet),
            lr = args.lr_encoder,
            betas=(0.9, 0.999))
    elif args.optimizer.lower() == 'radam':
        optimizer_unet = RAdam(
            group_weight(unet),
            lr=args.lr_encoder,
            betas=(0.9, 0.999))
    return [optimizer_unet]


def adjust_learning_rate(optimizers, cur_iter, args):
    scale_running_lr = 0.5*(1+math.cos(3.14159*(cur_iter)/args.num_epoch))
    args.running_lr_encoder = args.lr_encoder * scale_running_lr

    optimizer_unet = optimizers[0]
    for param_group in optimizer_unet.param_groups:
        param_group['lr'] = args.running_lr_encoder

def main(args):
    # Network Builders
    builder = ModelBuilder()

    unet = builder.build_unet(num_class=args.num_class,
        arch=args.unet_arch,
        weights=args.weights_unet)

    print("Froze the following layers: ")
    for name, p in unet.named_parameters():
        if p.requires_grad == False:
            print(name)
    print()
    #crit = nn.CrossEntropyLoss()
    crit = DualLoss(mode="train")

    segmentation_module = SegmentationModule(crit, unet)

    train_augs = Compose([PaddingCenterCrop(256), RandomHorizontallyFlip(), RandomVerticallyFlip(), RandomRotate(180)])
    test_augs = Compose([PaddingCenterCrop(256)])

    dataset_train = newTorontoData( 
            split='train',
            k_split=args.k_split,
            augmentations=train_augs)
    loader_train = data.DataLoader(
        dataset_train,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True)

    dataset_val = newTorontoData(
            split='val',
            k_split=args.k_split,
            augmentations=test_augs)
    
    loader_val = data.DataLoader(
         dataset_val,
         batch_size=1,
         shuffle=False,
         num_workers=int(args.workers),
         drop_last=True,
         pin_memory=True)

    # load nets into gpu
    if len(args.gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=args.gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit) if args.unet == False else (unet, crit)
    optimizers = create_optimizers(nets, args)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': [], 'jaccard': []}}
    best_val = {'epoch_1': 0, 'mIoU_1': 0,
                'epoch_2': 0, 'mIoU_2': 0,
                'epoch' : 0, 'mIoU': 0}

    max_iou = 0.

    # in_tensor = torch.zeros(1, 1, 10, 256, 256)
    # model = resnet3d.generate_model(50)
    # model(in_tensor)

    
    for epoch in range(args.start_epoch, args.num_epoch + 1):
        train(segmentation_module, loader_train, optimizers, history, epoch, args)
        iou, loss = eval(loader_val, segmentation_module, args, crit)
        # checkpointing
        if iou[-1] > max_iou:
            if epoch > 149:
                checkpoint(nets, history, args, epoch)
            max_iou = iou[-1]

        elif epoch % 30 == 0 and epoch > 149:
            checkpoint(nets, history, args, epoch)

    
    print('Training Done!')

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    DATA_ROOT = os.getenv('DATA_ROOT', '/PATH/TO/AC17/DATA')
    DATASET_NAME = "HCM_MRI_007"

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='baseline',
                        help="a name for identifying the model")
    parser.add_argument('--unet', default=True,
                        help="use unet?")
    parser.add_argument('--unet_arch', default='unet',
                        help="UNet architecture")
    parser.add_argument('--weights_unet', default='',
                        help="weights to finetune unet")

    # Path related arguments
    parser.add_argument('--data-root', type=str, default=DATA_ROOT)
    parser.add_argument('--result', default='./result')

    # optimization related arguments
    parser.add_argument('--gpus', default='0',
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=200, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=160, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--lr', default=0.0005, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--fix_bn', action='store_true',
                        help='fix bn params')

    # Data related argument
    parser.add_argument('--num_class', default=4, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of data loading workers')
    parser.add_argument('--dataset-name', type=str, default="AC17")
    parser.add_argument('--k_split', default=1)

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')

    parser.add_argument('--optimizer', default='sgd')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    if args.optimizer.lower() in ['sgd', 'adam', 'radam']:
        # Parse gpu ids
        all_gpus = parse_devices(args.gpus)
        all_gpus = [x.replace('gpu', '') for x in all_gpus]
        args.gpus = [int(x) for x in all_gpus]
        num_gpus = len(args.gpus)
        args.batch_size = num_gpus * args.batch_size_per_gpu
        args.gpu = 0
	
        args.lr_encoder = args.lr
        args.max_iters = args.num_epoch
        args.running_lr_encoder = args.lr_encoder

        # Model ID
        if args.unet ==False:
            args.id += '-' + args.arch_encoder
            args.id += '-' + args.arch_decoder
        else:
            args.id += '-' + str(args.unet_arch)

        args.id += '-ngpus' + str(num_gpus)
        args.id += '-batchSize' + str(args.batch_size)

        args.id += '-LR_unet' + str(args.lr_encoder)

        args.id += '-epoch' + str(args.num_epoch)

        print('Model ID: {}'.format(args.id))

        args.ckpt = os.path.join(args.ckpt, args.id)
        if not os.path.isdir(args.ckpt):
            os.makedirs(args.ckpt)

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        main(args)

    else:
        print("Invalid optimizer. Please try again with optimizer sgd, adam, or radam.")
