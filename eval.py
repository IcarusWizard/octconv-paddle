"""
This is adapted from 
https://github.com/facebookresearch/OctConv/blob/main/utils/gluon/score.py
"""

import argparse, time, logging, os, math

import numpy as np
import paddle
from paddle.io import DataLoader
import paddle.vision.transforms as T

from data import ImageNetDataset
from mobilenetv2 import mobilenet_v2_1125

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                        help='training and validation pictures to use.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--model', type=str, default='resnet101_v1d_hi',
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--ratio', type=float, default=0.,
                        help='percentage of the low frequency part')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--resume-params', type=str, default=None,
                        help='path of parameters to load from.')
    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    opt = parse_args()

    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(streamhandler)

    logger.info(opt)

    batch_size = opt.batch_size
    classes = 1000
    num_eval_samples = 50000

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    num_workers = opt.num_workers

    model_name = opt.model

    net = mobilenet_v2_1125(weight_file='weights/mobilenet_v2_1125_alpha-0.5.h5')
    net.eval()

    # create dataset and dataloader
    input_size = opt.input_size
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    transforms = T.Compose([
        T.Resize(resize),
        T.CenterCrop(input_size),
        T.Transpose([2, 0, 1]),
        T.Normalize([123.68, 116.779, 103.939], [58.393, 57.12, 57.375])
    ])
    val_dataset = ImageNetDataset(opt.data_dir, split='val', transforms=transforms)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=num_workers)

    acc_top1 = paddle.metric.Accuracy((1,))
    acc_top5 = paddle.metric.Accuracy((5,))

    def test(model, val_loader):
        model.eval()
        acc_top1.reset()
        acc_top5.reset()

        with paddle.no_grad():
            tic = time.time()
            for i, (images, labels) in enumerate(iter(val_loader)):
                outputs = model(images)
                top1_correct = acc_top1.compute(outputs, labels)
                acc_top1.update(top1_correct)
                top5_correct = acc_top5.compute(outputs, labels)
                acc_top5.update(top5_correct)

                if (i+1) % 50 == 0:
                    logger.info('[Batch %d][Image %d] validation: top1=%f top5=%f'%(i+1, (i+1)*batch_size, acc_top1.accumulate(), acc_top5.accumulate()))

        return acc_top1.accumulate(), acc_top5.accumulate(), time.time() - tic

    top1, top5, eclipse = test(net, val_loader)
    logger.info('speed: %d samples/sec\ttime cost: %fs'%(num_eval_samples/eclipse, eclipse))
    logger.info('validation: top1=%f top5=%f'%(top1, top5))
