from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Manager
from data_utils.DataLoader import MyKittiDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# tensorboard
writer_train = SummaryWriter('runs/mykitti_experiment6/train')
writer_test = SummaryWriter('runs/mykitti_experiment6/test')
writer_car_acc = SummaryWriter('runs/mykitti_experiment6/car_acc')
writer_cyclist_acc = SummaryWriter('runs/mykitti_experiment6/cyclist_acc')
writer_pedestrian_acc = SummaryWriter('runs/mykitti_experiment6/pedestrian_acc')
writer_dontcare_acc = SummaryWriter('runs/mykitti_experiment6/dontcare_acc')


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=250, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=256, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    return parser.parse_args()


def test(epoch, model, criterion, loader, num_class=4):
    mean_loss = 0
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, trans_feat = classifier(points)
        mean_loss += criterion(pred, target.long(), trans_feat).item()
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    # tensorboard
    mean_loss /= len(loader)
    writer_test.add_scalar('loss', mean_loss, epoch)

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    # class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
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

    '''DATA LOADING'''
    log_string('Load dataset ...')
    traindata_manager = Manager()
    testdata_manager = Manager()
    train_cache = traindata_manager.dict()
    test_cache = testdata_manager.dict()

    DATA_PATH = '/disk/users/sc468/no_backup/my_kitti'

    TRAIN_DATASET = MyKittiDataLoader(root=DATA_PATH, cache=train_cache, npoint=args.num_point, split='train',
                                      normal_channel=args.normal)
    TEST_DATASET = MyKittiDataLoader(root=DATA_PATH, cache=test_cache, npoint=args.num_point, split='test',
                                     normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_class = 4
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(num_class, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        print("Train cache size {}".format(len(train_cache)))
        print("Test cache size {}".format(len(test_cache)))

        mean_loss = 0
        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()

            # data augmentation
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.rotate_point_cloud_z(points[:, :, 0:3])
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.jitter_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            mean_loss += loss.item()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        # tensorboard
        mean_loss /= len(trainDataLoader)
        writer_train.add_scalar('loss', mean_loss, epoch)
        writer_train.add_scalar('overall accuracy', train_instance_acc, epoch)

        with torch.no_grad():
            instance_acc, class_acc = test(epoch, classifier.eval(), criterion, testDataLoader)
            mean_class_acc = np.mean(class_acc[:, 2])

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc

            if (mean_class_acc >= best_class_acc):
                best_class_acc = mean_class_acc
                best_epoch = epoch + 1

            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, mean_class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            log_string('Each Class Accuracy:')
            log_string('Car: %f' % class_acc[0, 2])
            log_string('Cyclist: %f' % class_acc[1, 2])
            log_string('Pedestrian: %f' % class_acc[2, 2])
            log_string('DontCare: %f' % class_acc[3, 2])
            print("")

            # tensorboard
            writer_test.add_scalar('overall accuracy', instance_acc, epoch)
            writer_car_acc.add_scalar('accuracy of each class', class_acc[0, 2], epoch)
            writer_cyclist_acc.add_scalar('accuracy of each class', class_acc[1, 2], epoch)
            writer_pedestrian_acc.add_scalar('accuracy of each class', class_acc[2, 2], epoch)
            writer_dontcare_acc.add_scalar('accuracy of each class', class_acc[3, 2], epoch)

            if (mean_class_acc >= best_class_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': mean_class_acc,
                    'each_class_acc': class_acc[:, 2],
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
