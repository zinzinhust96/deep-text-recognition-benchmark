import torch
from random import randint
import uuid
import os
from PIL import Image, ImageOps, ImageDraw, ImageFont

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def crop_image_with_pad(img, min_pad=5, max_pad=10):
    w, h = img.size
    top_pad = randint(min_pad, max_pad)
    bottom_pad = randint(min_pad, max_pad)
    left_pad = randint(min_pad, max_pad)
    right_pad = randint(min_pad, max_pad)
    area = (max_pad - left_pad, max_pad - top_pad, w - (max_pad - right_pad), h - (max_pad - bottom_pad))
    cropped_img = img.crop(area)
    return cropped_img

def save_prediction_result(img, img_path, pred, opt):
    # add bottom border to image
    img = ImageOps.expand(img, border=(0, 0, 0, 60))

    fontpath = "./fonts/AndikaNewBasic-R.ttf"
    font = ImageFont.truetype(fontpath, 40)
    draw = ImageDraw.Draw(img)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((0, img.size[1] - 60), f'{pred}', font = font, fill = (255,255,255))

    #Save image
    img.save(os.path.join(opt.save_results, img_path.split('/')[-1]))

def save_prediction_results_with_gt(img, pred, gt, confidence_score, folder_to_save, saved_img_name):
    # add bottom border to image
    img = ImageOps.expand(img, border=(0, 0, 0, 90))

    fontpath = "./fonts/AndikaNewBasic-R.ttf"
    font = ImageFont.truetype(fontpath, 20)
    draw = ImageDraw.Draw(img)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((0, img.size[1] - 30), f'{confidence_score}', font = font, fill = (255,255,255))
    draw.text((0, img.size[1] - 60), f'{pred}', font = font, fill = (255,0,0))
    draw.text((0, img.size[1] - 90), f'{gt}', font = font, fill = (0,255,0))

    with open(os.path.join(os.path.join('result', folder_to_save, 'log_prediction.txt')), 'a') as fopen:
        fopen.write('{}\t{}\t{}\n'.format(saved_img_name, gt, pred))

    save_path = os.path.join('result', folder_to_save, 'result')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img.save(os.path.join(save_path, '{}.jpg'.format(saved_img_name)))

if __name__ == '__main__':
    ctc = AttnLabelConverter('0123456789abcdefghijklmnopqrstuvwxyz')
    t, l = ctc.encode(['miiinh'])
    print(t, l)
    print(ctc.decode(t, l))