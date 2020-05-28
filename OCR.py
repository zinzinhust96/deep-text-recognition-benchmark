import string
import argparse
import os
import cv2
import re
import time
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

import glob
import math

from utils import CTCLabelConverter, AttnLabelConverter
from model import Model
from dataset import NormalizePAD

def align_collate_image(images, opt):
    if opt.PAD:  # same concept with 'Rosetta' paper
        resized_max_w = opt.imgW
        input_channel = 3 if images[0].mode == 'RGB' else 1
        transform = NormalizePAD((input_channel, opt.imgH, resized_max_w))

        resized_images = []
        for ieee, image in enumerate(images):
            w, h = image.size
            ratio = w / float(h)
            if math.ceil(opt.imgH * ratio) > opt.imgW:
                resized_w = opt.imgW
            else:
                resized_w = math.ceil(opt.imgH * ratio)
            
            resized_image = image.resize((resized_w, opt.imgH), Image.BICUBIC) #ANTIALIAS
            resized_images.append(transform(resized_image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

    else:
        transform = ResizeNormalize((opt.imgW, opt.imgH))
        image_tensors = [transform(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

    return image_tensors

class OCR():
    def __init__(self, opt, cuda):
        if cuda:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')

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
        best_model_names = ['best_norm_ED.pth', 'best_accuracy.pth', 'best_valid_loss.pth', 'TPS-ResNet-BiLSTM-CTC.pth']

        if any(model_name in opt.saved_model for model_name in best_model_names):
            model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        else:
            checkpoint = torch.load(opt.saved_model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()

        ###
        self.converter = converter
        self.model = model
        self.opt = opt
        self.device = device

    def recognize(self, images):
        # process image
        image_tensors = align_collate_image(images, self.opt)
        batch_size = image_tensors.size(0)
        image = image_tensors.to(self.device)

        # For max length prediction
        length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(self.device)
        text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

        if 'CTC' in self.opt.Prediction:
            preds = self.model(image, text_for_pred)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)

        else:
            preds = self.model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        ocr_results = []
        for pred_word, pred_max_prob in zip(preds_str, preds_max_prob):
            if 'Attn' in self.opt.Prediction:
                pred_EOS = pred_word.find('[s]')
                pred_word = pred_word[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            
            ocr_results.append([pred_word, confidence_score])
        
        return ocr_results
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default='/project/UBD_OCR/src/deep-text-recognition-benchmark/demo_image/exp', help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default='/projects/OCR/deep-text-recognition-benchmark/saved_models/TPS-ResNet-BiLSTM-CTC-Seed2003/best_valid_loss.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=64, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=512, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789./-', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='CTC', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    # some of the settings
    opt.PAD = True
    opt.sensitive = True
    opt.imgW = 512
    opt.imgH = 64
    opt.character = "0123456789aáàãạảăắằẵặẳâấầẫậẩbcdđeéèẽẹẻêếềễệểfghiíìĩịỉjklmnoóòõọỏôốồỗộổơớờỡợởpqrstuúùũụủưứừữựửvwxyýỳỹỵỷzAÁÀÃẠẢĂẮẰẴẶẲÂẤẦẪẬẨBCDĐEÉÈẼẸẺÊẾỀỄỆỂFGHIÍÌĨỊỈJKLMNOÓÒÕỌỎÔỐỒỖỘỔƠỚỜỠỢỞPQRSTUÚÙŨỤỦƯỨỪỮỰỬVWXYÝỲỸỴỶZ\/?.,:;()!'/-"

    ocr = OCR(opt, cuda=True)

    images = []
    img = Image.open('/hdd/DATA/IDCard_words/19_03_val/img/24_6_01668507120_truoc_2_0.jpg').convert('L')
    images.append(img)
    img = Image.open('/hdd/DATA/IDCard_words/19_03_val/img/24_21_78393362797_truoc_0_0.jpg').convert('L')
    images.append(img)

    results = ocr.recognize(images)
    print(results)
