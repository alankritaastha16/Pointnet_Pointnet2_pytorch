"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
#from  data_utils import dataloader as data
from data_utils.ShapeNetDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--num_itr', default=3, type=int, help='number of iterations for inference')
    #parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2500, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--datadir', type=str, default='/data/pointclouds/ShapeNet', help='decay rate for lr decay')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = args.datadir + '/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
    #TRAIN_DATASET = data.ICCV17ShapeNet(args.datadir, 'train', None, 'part')
    #TEST_DATASET = data.ICCV17ShapeNet(args.datadir, 'test', None, 'part')
    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
   # TRAIN_DATASET = torch.utils.data.Subset(TRAIN_DATASET, range(0, 500))
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 16
    num_part = 50
    num_itr=args.num_itr
    input_size = num_part

    '''MODEL LOADING'''
    MODELS = [importlib.import_module(args.model) for i in range(num_itr)]
    #print(MODELS)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    classifiers = [MODELS[i].get_model(num_part, input_size, normal_channel=args.normal).to(device) for i in range(num_itr)]
    criterions = [MODELS[i].get_loss().to(device) for i in range(num_itr)]
    [classifiers[i].apply(inplace_relu) for i in range(num_itr)]

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = [torch.load(str(exp_dir) + '/checkpoints/model-{i}.pth') for i in range(num_itr)]
        start_epoch = checkpoint[num_itr-1]['epoch']+1
        [classifiers[i].load_state_dict(checkpoint['model_state_dict'] for i in range(num_itr))]
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifiers = [classifiers[i].apply(weights_init) for i in range(num_itr)]

    if args.optimizer == 'Adam':
        optimizers = [torch.optim.Adam(
            classifiers[i].parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        ) for i in range(num_itr)]
    else:
        optimizers = [torch.optim.SGD(classifiers[i].parameters(), lr=args.learning_rate, momentum=0.9) for i in range(num_itr)]

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for i in range(num_itr):
            for param_group in optimizers[i].param_groups:
                param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifiers = [classifiers[i].apply(lambda x: bn_momentum_adjust(x, momentum)) for i in range(num_itr)]
        classifiers = [classifiers[i].train() for i in range(num_itr)]
        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total = len(trainDataLoader), smoothing=0.9):
            prev_output = torch.zeros((points.shape[0], num_part, points.shape[1]), device=device)
            mean_correct = [[] for _ in range(num_itr)]
           # print('prev_output:',prev_output.shape)
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
            points = points.transpose(2, 1)
           # print('points:', points.shape)
            for j in range(num_itr):
               # print(j)
                optimizers[j].zero_grad()
                _points = torch.cat((points, prev_output), 1)
                seg_pred, trans_feat = classifiers[j](_points, to_categorical(label, num_classes))
                prev_output = seg_pred.transpose(2, 1).detach()  # detach() so that prev_output is now a constant
                #print('prev_output:',prev_output.shape)
                seg_pred = seg_pred.contiguous().view(-1, num_part)
                # print('seg_pred:', seg_pred.shape)
                target = target.view(-1, 1)[:, 0]
                pred_choice = seg_pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                mean_correct[j].append(correct.item() / (args.batch_size * args.npoint))
                loss = criterions[j](seg_pred, target, trans_feat)
                loss.backward()  # retain_graph=True
                optimizers[j].step()
        #print('mean_correct:',mean_correct)
        for i in range(num_itr):
            train_instance_acc = np.mean(mean_correct[i])
            log_string('Train accuracy model-%d is: %.5f' % (i , train_instance_acc))
        if epoch % 1 == 0:
            logger.info('Save model...')
            for i in range(num_itr):
                savepath = str(checkpoints_dir) + f'/model-{i}.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'model_state_dict': classifiers[i].state_dict(),
                    'optimizer_state_dict': optimizers[i].state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')

            '''
            with torch.no_grad():
                test_metrics = {}
                total_correct = 0
                total_seen = 0
                total_seen_class = [0 for _ in range(num_part)]
                total_correct_class = [0 for _ in range(num_part)]
                shape_ious = {cat: [] for cat in seg_classes.keys()}
                seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

                for cat in seg_classes.keys():
                    for label in seg_classes[cat]:
                        seg_label_to_cat[label] = cat

                classifier = classifiers[i].eval()

                for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                    cur_batch_size, NUM_POINT, _ = points.size()
                    points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
                    points = points.transpose(2, 1)
                    seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    target = target.cpu().data.numpy()

                    for i in range(cur_batch_size):
                        cat = seg_label_to_cat[target[i, 0]]
                        logits = cur_pred_val_logits[i, :, :]
                        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                    correct = np.sum(cur_pred_val == target)
                    total_correct += correct
                    total_seen += (cur_batch_size * NUM_POINT)

                    for l in range(num_part):
                        total_seen_class[l] += np.sum(target == l)
                        total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                    for i in range(cur_batch_size):
                        segp = cur_pred_val[i, :]
                        segl = target[i, :]
                        cat = seg_label_to_cat[segl[0]]
                        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                        for l in seg_classes[cat]:
                            if (np.sum(segl == l) == 0) and (
                                    np.sum(segp == l) == 0):  # part is not present, no prediction as well
                                part_ious[l - seg_classes[cat][0]] = 1.0
                            else:
                                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                    np.sum((segl == l) | (segp == l)))
                        shape_ious[cat].append(np.mean(part_ious))

                all_shape_ious = []
                for cat in shape_ious.keys():
                    for iou in shape_ious[cat]:
                        all_shape_ious.append(iou)
                    shape_ious[cat] = np.mean(shape_ious[cat])
                mean_shape_ious = np.mean(list(shape_ious.values()))
                test_metrics['accuracy'] = total_correct / float(total_seen)
                test_metrics['class_avg_accuracy'] = np.mean(
                    np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
                for cat in sorted(shape_ious.keys()):
                    log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
                test_metrics['class_avg_iou'] = mean_shape_ious
                test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

            log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
                epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
            if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'train_acc': train_instance_acc,
                    'test_acc': test_metrics['accuracy'],
                    'class_avg_iou': test_metrics['class_avg_iou'],
                    'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizers[j].state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')

            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
            if test_metrics['class_avg_iou'] > best_class_avg_iou:
                best_class_avg_iou = test_metrics['class_avg_iou']
            if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
                best_inctance_avg_iou = test_metrics['inctance_avg_iou']
            log_string('Best accuracy is: %.5f' % best_acc)
            log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
            log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
            '''
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
