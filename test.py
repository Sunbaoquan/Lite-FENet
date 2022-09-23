import math
import os
import time
import cv2
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from util import dataset, transform, config
from util.util import AverageMeter, intersectionAndUnionGPU, check_makedirs, get_logger, set_seed
from model.Lite_FENet import Lite_FENet

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_resnet50.yaml', help='config file path')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in [0, 1, 2, 3, 999]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)  # gpus for trainint
    print("The ngpus_per_node is: ", args.ngpus_per_node)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args)


def main_worker(argss):
    global args, logger
    args = argss
    check_makedirs(args.save_path)
    logger = get_logger(args.save_path + '/test-{}shot.log'.format(args.shot))

    model = Lite_FENet(layers=args.layers, classes=2, zoom_factor=8,
                       criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d,
                       backbone_pretrained=True,
                       shot=args.shot, scales=args.scales, vgg=args.vgg)

    logger.info("=> creating model ...")
    print(args)

    if args.manual_seed is not None:
        set_seed(args.manual_seed, deterministic=False)
    model = model.cuda()
    val_num = 5
    seed_array = np.array([320, 321, 322, 323, 324])
    MIoU_array = np.zeros(val_num)

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> No weight found at '{}'".format(args.weight))

    # ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])

    val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, data_list=args.val_list,
                               transform=val_transform, mode='val',
                               use_coco=args.use_coco, use_split_coco=args.use_split_coco)

    val_loader = DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                            num_workers=args.workers, pin_memory=True, sampler=None)

    for val_id in range(val_num):
        val_seed = seed_array[val_id]
        print('Val: [{}/{}] \t Seed: {}'.format(val_id + 1, val_num, val_seed))
        loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model, val_seed)
        MIoU_array[val_id] = class_miou

    print('MIoU:  {}'.format(np.round(MIoU_array, 4)))
    print('-' * 43)
    print('Mean_mIoU: {:.4f}'.format(MIoU_array.mean()))
    with open(args.save_path + '/test-{}shot.log'.format(args.shot), 'a') as f:
        f.write('MIoU: {}\n'.format(np.round(MIoU_array, 4)))
        f.write('-' * 43 + '\n')
        f.write('Mean_mIoU: {:.4f}'.format(MIoU_array.mean()))



def validate(val_loader, model, val_seed):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    if args.use_coco:
        split_gap = 20
        test_num = 20000
    else:
        split_gap = 5
        test_num = 1000

    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap
    set_seed(val_seed, False)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    model.eval()

    end = time.time()
    assert test_num % args.batch_size_val == 0
    iter_num = 0
    total_time = 0

    print("Val dataLoader len is [{}]".format(len(val_loader)))
    db_epoch = math.ceil(test_num/(len(val_loader)-args.batch_size_val))

    for e in range(db_epoch):
        for i, (input, target, support_input, support_mask, subcls, ori_label) in enumerate(val_loader):
            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)
            support_input = support_input.cuda(non_blocking=True)
            support_mask = support_mask.cuda(non_blocking=True)
            start_time = time.time()
            output = model(s_x=support_input, s_y=support_mask, x=input, y=target)
            total_time = total_time + 1
            model_time.update(time.time() - start_time)

            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda() * 255
                backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()

            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            loss = criterion(output, target)

            n = input.size(0)
            loss = torch.mean(loss)
            output = output.max(1)[1]

            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

            subcls = subcls[0].cpu().numpy()[0]

            class_intersection_meter[subcls] += intersection[1]
            class_union_meter[subcls] += union[1]

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % (test_num / 100) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})'
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})'
                            'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              loss_meter=loss_meter,
                                                              accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)  # FB-IoU
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(split_gap):
        logger.info('Class_{} Result: IoU {:.4f}.'.format(i + 1, class_iou_class[i]))

    logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.classes):
        logger.info('Class_{} Result: FB-Iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('Avg Inference Time: {:.4f}, count: {}'.format(model_time.avg, test_num))

    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou


if __name__ == '__main__':
    main()
