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
import torch.nn.functional as F

from PIL import Image, ImageOps, ImageDraw, ImageFont
from utils import CTCLabelConverter, AttnLabelConverter, save_prediction_result
from dataset import RawDataset, AlignCollate, CollateFn
from model import Model
from date_extractor.extractor import extract_dmy_from_text
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
    best_model_names = ['best_norm_ED.pth', 'best_accuracy.pth', 'best_valid_loss.pth', 'TPS-ResNet-BiLSTM-CTC.pth', 'None-ResNet-None-CTC.pth']

    if any(model_name in opt.saved_model for model_name in best_model_names):
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    else:
        checkpoint = torch.load(opt.saved_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    # AlignCollate_demo = CollateFn(imgH=32)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=1,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    log = open(f'./log_demo_result.txt', 'a')
    dashed_line = '-' * 80
    head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
    
    print(f'{dashed_line}\n{head}\n{dashed_line}')
    log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')
    with torch.no_grad():
        for index, (image_tensors, _, image_path_list) in enumerate(demo_loader):
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred, index=index)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                # preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                # preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            # decode sequence of token
            if 'CTC' in opt.Prediction:
                preds_str, preds_score, raws_str = converter.decode_with_threshold(preds_index.data, preds_size.data, preds_max_prob)
                # print('raws_str', np.array(raws_str).shape, raws_str)

            else:
                preds_str = converter.decode(preds_index, length_for_pred)

            for img_path, pred, pred_score, raw_str, pred_max_prob in zip(image_path_list, preds_str, preds_score, raws_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_path:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_path:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

                '''
                if len(raw_str) == len(pred_max_prob):
                    print('===== confident score for each token =====')
                    for token, score in zip(raw_str, pred_max_prob):
                        print(token, '\t', score.item())
                    print('\n')            
                # '''

                # '''
                if len(pred) == len(pred_score):
                    print('===== confident score for final set of token =====')
                    for token, score in zip(pred, pred_score):
                        print(token, '\t', score.item())
                    print('\n')            
                # '''

                # write results image
                save_prediction_result(Image.open(img_path), img_path, pred, opt)

    log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--save_results', required=True, help="path to saved results")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=256, help='the width of the input image')
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

    # if not os.path.exists(opt.save_results):
    #     os.makedirs(opt.save_results)

    # load custom character list
    # f=open("char_list.txt", "r")
    # opt.character = f.read()

    """ vocab / character number configuration """
    # if opt.sensitive:
    #     opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    print('PADDING: ', opt.PAD)

    demo(opt)
