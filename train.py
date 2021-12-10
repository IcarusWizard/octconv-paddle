"""
This is adapted from 
https://github.com/facebookresearch/OctConv/blob/main/utils/gluon/train_imagenet.py
"""

import argparse, time, logging, os, math

import paddle
import paddle.distributed as dist

from paddle.io import DataLoader
import paddle.vision.transforms as T
import numpy as np

from data import ImageNetDataset
from mobilenetv2 import mobilenet_v2_1125
from utils import setup_seed

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                        help='training and validation pictures to use.')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=150,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.00004,
                        help='weight decay rate. default is 0.00004.')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='number of warmup epochs.')
    parser.add_argument('--last-gamma', action='store_true',
                        help='whether to init gamma of the last BN layer in each bottleneck to 0.')
    parser.add_argument('--model', type=str, default='mobilenet_v2_1125',
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='percentage of the low frequency part')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='use label smoothing or not in training. default is false.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--final-drop', type=float, default=0.,
                        help='whether to use dropout before the laster classifier. default: 0 (disable).')
    parser.add_argument('--save-frequency', type=int, default=10,
                        help='frequency of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-epoch', type=int, default=0,
                        help='epoch to resume training from.')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--resume-states', type=str, default='',
                        help='path of trainer state to load from.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--logging-file', type=str, default='train_imagenet.log',
                        help='name of training log file')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    setup_seed(opt.seed)

    filehandler = logging.FileHandler(opt.logging_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    rank = paddle.distributed.get_rank()
    if rank == 0: logger.info(opt)

    batch_size = opt.batch_size
    classes = 1000

    num_gpus = opt.num_gpus
    init_parallel = opt.num_gpus > 1
    if init_parallel: dist.init_parallel_env()
    batch_size *= max(1, num_gpus)
    num_workers = opt.num_workers

    # create dataset and dataloader
    input_size = opt.input_size
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))

    train_transforms = T.Compose([
        T.RandomResizedCrop(input_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.4, 0.4, 0.4),
        # TODO: Add AlexNet PCA-based random lighting with p=0.1 
        T.Transpose([2, 0, 1]),
        T.Normalize([123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
    ])
    train_dataset = ImageNetDataset(opt.data_dir, split='train', transforms=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    val_transforms = T.Compose([
        T.Resize(resize),
        T.CenterCrop(input_size),
        T.Transpose([2, 0, 1]),
        T.Normalize([123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
    ])
    val_dataset = ImageNetDataset(opt.data_dir, split='val', transforms=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # create model and optimizor
    model_name = opt.model

    kwargs = {
        'classes': classes,
        'final_drop': opt.final_drop,
        'zero_last_gamma' : opt.last_gamma,
    }

    # if opt.dtype != 'float32':
    #     optimizer_params['multi_precision'] = True

    if model_name == 'mobilenet_v2_1125':
        model = mobilenet_v2_1125(ratio=opt.ratio, **kwargs)
    else:
        raise NotImplementedError('Unknown model name!')
    
    if init_parallel: model = paddle.DataParallel(model)

    lr_decay_period = opt.lr_decay_period
    if opt.lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]

    num_training_samples = len(train_dataset) 
    steps_per_epoch = num_training_samples // batch_size
    lr_decay_steps = [e * steps_per_epoch for e in lr_decay_epoch]
    
    if opt.lr_mode == 'step':
        lr_scheduler = paddle.optimizer.lr.MultiStepDecay(opt.lr, milestones=lr_decay_steps, gamma=opt.lr_decay)
    elif opt.lr_mode == 'cosine':
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(opt.lr, opt.num_epochs * steps_per_epoch)
    lr_scheduler = paddle.optimizer.lr.LinearWarmup(lr_scheduler, warmup_steps=opt.warmup_epochs * steps_per_epoch, start_lr=opt.warmup_lr, end_lr=opt.lr)
    
    normal_parameters = []
    no_decay_parameters = []
    named_parameters = dict(model.named_parameters())
    for name, param in named_parameters.items():
        if name.endswith('.bias'):
            no_decay_parameters.append(param)
        elif name.endswith('.weight'):
            if (name[:-6] + '_mean') in  named_parameters.keys():
                no_decay_parameters.append(param)
            else:
                normal_parameters.append(param)
        else:
            normal_parameters.append(param)
    parameters = [
        {
            'params' : normal_parameters
        },
        {
            'params' : no_decay_parameters,
            'weight_decay' : (1 - int(opt.no_wd)) * opt.wd
        }
    ]

    optim = paddle.optimizer.Momentum(learning_rate=lr_scheduler, momentum=opt.momentum, parameters=parameters, use_nesterov=True, weight_decay=opt.wd)

    loss_fn = paddle.nn.loss.CrossEntropyLoss(soft_label=opt.label_smoothing)
    train_metric = paddle.metric.Accuracy((1,))
    acc_top1 = paddle.metric.Accuracy((1,))
    acc_top5 = paddle.metric.Accuracy((5,))

    save_frequency = opt.save_frequency
    if opt.save_dir and save_frequency:
        save_dir = opt.save_dir
        if not os.path.exists(save_dir): os.makedirs(save_dir)
    else:
        save_dir = ''
        save_frequency = 0

    def test():
        model.eval()
        acc_top1.reset()
        acc_top5.reset()

        with paddle.no_grad():
            for i, (images, labels) in enumerate(iter(val_loader)):
                outputs = model(images)
                top1_correct = acc_top1.compute(outputs, labels)
                acc_top1.update(top1_correct)
                top5_correct = acc_top5.compute(outputs, labels)
                acc_top5.update(top5_correct)

        return 1 - acc_top1.accumulate(), 1 - acc_top5.accumulate()

    def train():
        model.train()

        best_val_score = 1
        start_time = time.time()

        for epoch in range(opt.resume_epoch, opt.num_epochs):
            tic = time.time()

            train_metric.reset()
            btic = time.time()
            losses = []
            for i, (data, labels) in enumerate(iter(train_loader)):
                if opt.label_smoothing:
                    onthot_labels = paddle.nn.functional.one_hot(labels, classes)
                    _labels = paddle.nn.functional.label_smooth(onthot_labels, epsilon=0.1)
                else:
                    _labels = labels

                logits = model(data)
                loss = loss_fn(logits, _labels)
                optim.clear_grad()
                loss.backward()
                optim.step()
                lr_scheduler.step()

                losses.append(loss.numpy()[0])

                train_metric.update(train_metric.compute(logits, labels))

                if rank == 0 and opt.log_interval and not (i+1)%opt.log_interval:
                    logger.info('Epoch[%d] Batch [%d]\tTime used %f (s)\tSpeed: %f samples/sec\tloss=%f\t%s=%f\tlr=%f'%(
                                epoch, i, time.time() - start_time, batch_size*opt.log_interval/(time.time()-btic), 
                                np.mean(losses), 'acc', train_metric.accumulate(), lr_scheduler.get_lr()))
                    btic = time.time()

            throughput = int(batch_size * i /(time.time() - tic))

            err_top1_val, err_top5_val = test()

            if rank == 0:
                logger.info('[Epoch %d] training: %s=%f'%(epoch, 'acc', train_metric.accumulate()))
                logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f'%(epoch, throughput, time.time()-tic))
                logger.info('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))

            if rank == 0 and err_top1_val < best_val_score:
                best_val_score = err_top1_val
                paddle.save(model.state_dict(), '%s/%.4f-imagenet-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))
                paddle.save(optim.state_dict(), '%s/%.4f-imagenet-%s-%d-best.states'%(save_dir, best_val_score, model_name, epoch))

            if rank == 0 and save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
                paddle.save(model.state_dict(), '%s/imagenet-%s-%d.params'%(save_dir, model_name, epoch))
                paddle.save(optim.state_dict(), '%s/imagenet-%s-%d.states'%(save_dir, model_name, epoch))

        if rank == 0 and save_frequency and save_dir:
            paddle.save(model.state_dict(), '%s/imagenet-%s-%d.params'%(save_dir, model_name, opt.num_epochs-1))
            paddle.save(optim.state_dict(), '%s/imagenet-%s-%d.states'%(save_dir, model_name, opt.num_epochs-1))

    train()

if __name__ == '__main__':
    main()
