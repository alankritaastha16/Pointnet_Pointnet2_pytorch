"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    #parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--datadir', type=str, default= '/data/pointclouds/s3dis-yanx27', help='path to data directory')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--num_itr', type=int, default=5, help='number of iterations for iterative inference')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    #parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()

# def add_vote(vote_label_pool, point_idx, pred_label, weight):
#     B = pred_label.shape[0]
#     N = pred_label.shape[1]
#     for b in range(B):
#         for n in range(N):
#             if weight[b, n] != 0 and not np.isinf(weight[b, n]):
#                 vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
#     return vote_label_pool

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    experiment_dir = 'log/sem_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)


    root = args.datadir+ +'s3dis/stanford_indoor3d/'
    NUM_CLASSES = 13
    NUM_POINT = args.num_point
    BATCH_SIZE = args.batch_size
    input_size=NUM_CLASSES
    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    
    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES,input_size).to(device)
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    '''Evaluate on chopped scenes'''
    with torch.no_grad():
        num_batches = len(testDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        labelweights = np.zeros(NUM_CLASSES)
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().to(device), target.long().to(device)
            points = points.transpose(2, 1)
            prev_output = torch.zeros((points.shape[0], NUM_CLASSES, points.shape[2]), device=device)
            for j in range(args.num_itr):
                _points = torch.cat((points, prev_output), 1)
                #print('points:', _points.shape)
                seg_pred, trans_feat = classifier(_points)
                prev_output = seg_pred.transpose(2, 1)
            #seg_pred, trans_feat = classifier(points)
            pred_val = seg_pred.contiguous().cpu().data.numpy()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            #loss = criterion(seg_pred, target, trans_feat, weights)
            #loss_sum += loss
            pred_val = np.argmax(pred_val, 2)
            correct = np.sum((pred_val == batch_label))
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
            labelweights += tmp

            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
        #log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('eval point avg class IoU: %f' % (mIoU))
        log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
        log_string('eval point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                total_correct_class[l] / float(total_iou_deno_class[l]))

        log_string(iou_per_class_str)

if __name__ == '__main__':
    args = parse_args()
    main(args)
