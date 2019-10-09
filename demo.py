import string
import argparse
import os
import cv2
import re
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    # with torch.no_grad():
    for image_tensors, image_path_list in demo_loader:
        # batch_size = image_tensors.size(0)
        batch_size = 1
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        preds_str = []
        batch_time = []
        if 'CTC' in opt.Prediction:
            # preds = model(image, text_for_pred).log_softmax(2)

            # # Select max probabilty (greedy decoding) then decode index to character
            # preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # _, preds_index = preds.permute(1, 0, 2).max(2)
            # preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            # preds_str = converter.decode(preds_index.data, preds_size.data)

            # predict on each image
            for i in range(0, image.shape[0]):
                start_time = time.time()
                one_img = image[i]
                one_img = one_img[None, :, :, :]
                pred = model(one_img, text_for_pred).log_softmax(2)
                # print('time: ', time.time() - start_time)

                # Select max probabilty (greedy decoding) then decode index to character
                pred_size = torch.IntTensor([pred.size(1)] * batch_size)
                _, pred_index = pred.permute(1, 0, 2).max(2)
                pred_index = pred_index.transpose(1, 0).contiguous().view(-1)
                pred_str = converter.decode(pred_index.data, pred_size.data)

                # add to list
                preds_str.append(pred_str[0])
                batch_time.append(time.time() - start_time)


        else:
            # preds = model(image, text_for_pred, is_train=False)

            # # select max probabilty (greedy decoding) then decode index to character
            # _, preds_index = preds.max(2)
            # preds_str = converter.decode(preds_index, length_for_pred)
            # predict on each image
            for i in range(0, image.shape[0]):
                start_time = time.time()
                one_img = image[i]
                one_img = one_img[None, :, :, :]
                pred = model(one_img, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, pred_index = pred.max(2)
                pred_str = converter.decode(pred_index, length_for_pred)

                # add to list
                preds_str.append(pred_str[0])
                batch_time.append(time.time() - start_time)


        print('-' * 80)
        print('image_path\tpredicted_labels\tElapsed time')
        print('-' * 80)
        for img_name, pred, t in zip(image_path_list, preds_str, batch_time):
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

            print(f'{img_name}\t{pred}\t{t}')

    print('Total time: ', np.sum(batch_time[1:]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    print('PADDING: ', opt.PAD)

    demo(opt)
