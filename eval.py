import argparse
import os
# from torchaudio.models.decoder import ctc_decoder
import json
import torch
import logging
import pathlib
import traceback
from FOTS.utils.util import keys
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
from FOTS.model.model import FOTSModel
from FOTS.utils.bbox import Toolbox
import torch.nn as nn
import easydict
from FOTS.data_loader.data_module import ICDARDataModule
import cv2
import numpy as np
torch.cuda.empty_cache()
torch.cuda.memory_summary()
logging.basicConfig(level=logging.DEBUG, format='')
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels,blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank
        # self.tree = tree

    def forward(self, emission: torch.Tensor,to_string=True):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        oute =''
        index = torch.argmax(emission, dim=-1)
        print([self.labels[inde] for inde in index])
        for inde in emission:
            # list_num = list_cha(oute,self.tree,self.labels)
            inde = inde#+ list_num*0.4
            inde = torch.argmax(inde, dim=-1)  # [num_seq,]
            # inde = torch.unique_consecutive(inde, dim=-1)
            # predict(oute,'',self.tree)
            if inde != self.blank:
                oute += self.labels[inde.data.cpu().numpy()-1 if inde.data.cpu().numpy()>1 else 0]
            else:
                oute += ' '
        last = ''
        final = ''
        for st in oute:
            if st!= last:
                final += st
            last = st
        final = final.strip(' ')
        print(final)
        if to_string:
            return final
        else:
            return index
decoder_ctc =  GreedyCTCDecoder(keys)


# greedy_decoder = GreedyCTCDecoder(tokens)

def load_model(model_path, config ,resume = True):
    model = FOTSModel(config)

    if config.data_loader.dataset == 'synth800k':
        data_module = SynthTextDataModule(config)
    else:
        data_module = ICDARDataModule(config)

    root_dir = str(pathlib.Path(config.trainer.save_dir).absolute() / config.name)
    if resume:
        assert pathlib.Path(config.pretrain).exists()
        resume_ckpt = config.pretrain
        logger.info('Resume training from: {}'.format(config.pretrain))
    else:
        if config.pretrain:
            assert pathlib.Path(config.pretrain).exists()
            logger.info('Finetune with: {}'.format(config.pretrain))
            model.load_from_checkpoint(config.pretrain, config=config, map_location='cpu')
            resume_ckpt = None
        else:
            resume_ckpt = None
    if os.path.exists(resume_ckpt):
        checkpoint = torch.load(resume_ckpt)
        model.load_state_dict(checkpoint)
    return model


def main(args:argparse.Namespace):
    model_path = args.model
    input_dir = args.input_dir
    output_dir = args.output_dir
    with_image = True if output_dir else False
    with_gpu = True if torch.cuda.is_available() else False

    config = json.load(open(args.config))
    #with_gpu = False


    config = easydict.EasyDict(config)
    # model = FOTSModel.load_from_checkpoint(checkpoint_path=model_path,
    #                                        map_location='cpu', config=config)
    model = load_model(model_path,config,True)
    model = model.to('cuda:0')
    model.eval()
    txt_path = "combine_cccd_collect.txt"
    output_dir = None
    # with_image = False
    for image_fn in input_dir.glob('*.jpg'):
        # try:
        read_done = 0
        with torch.no_grad():
            img = cv2.imread(str(image_fn))
        # print(image_fn)
            ploy, im, pred,loss,vertices = Toolbox.predict(image_fn, model,txt_path, with_image, output_dir, with_gpu=True)
            print(loss)
            # print(len(ploy))
        # with open("F:/project_2/New_folder/combile_word.txt",'r',encoding='utf-8') as f:
        decoder_ctc =  GreedyCTCDecoder(keys,blank= 236)
        string_text = ''
        for id in range(pred[0].shape[1]):
            result_text = (pred[0][:,id,:]) #nn.Softmax(dim=1)
            # plt.imshow(result_text.data.cpu().numpy().transpose())
            # plt.show()
            str_text = decoder_ctc.forward(torch.tensor(result_text),to_string=True)
            if np.min(ploy[id]) >0:
                plt.imshow(im[ploy[id][0,1]:ploy[id][2,1],ploy[id][0,0]:ploy[id][2,0],:])
                plt.title(str_text)
                plt.show()
                string_text += str_text
                string_text += ' '
        print(string_text)


        # with open("combine_cccd_collect.txt",'r',encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip('\n').split('*')
        #         x1,y1,x2,y2,x3,y3, x4,y4 = np.int32(line[2:])
        #         if line[0].split('/')[-1] == str(image_fn).split('\\')[-1]:
        #             img = cv2.polylines(img,[np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]],np.int32)],True,color= [255,0,0],thickness= 2)
        #             read_done = 1
        #         elif read_done ==1:
        #             cv2.imshow('img',img)
        #             cv2.waitKey()
        #             break


        # except Exception as e:
        #     traceback.print_exc()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='epoch=38-step=935_7_9.ckpt', type=pathlib.Path,
                        help='path to model')
    parser.add_argument('-o', '--output_dir', default='F:/project_2/New_folder/output', type=pathlib.Path,
                        help='output dir for drawn images')
    parser.add_argument('-i', '--input_dir', default='F:/project_2/New_folder/test/', type=pathlib.Path, required=False,
                        help='dir for input images')
    parser.add_argument('-c', '--config', default='pretrain.json', type=str,
                        help='config file path (default: None)')
    args = parser.parse_args('')
    main(args)









