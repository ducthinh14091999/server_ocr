import argparse
import os

import json
import torch
import logging
import pathlib
import traceback

from pytorch_lightning import Trainer

from .FOTS.model.model import FOTSModel
from .FOTS.utils.bbox import Toolbox

import easydict
import cv2
import numpy as np
torch.cuda.empty_cache()
torch.cuda.memory_summary()
logging.basicConfig(level=logging.DEBUG, format='')


def load_model(config ,resume = True):
    model = FOTSModel(config)
    if resume:
        assert pathlib.Path(config.pretrain).exists()
        resume_ckpt = config.pretrain
        # logger.info('Resume training from: {}'.format(config.pretrain))
    else:
        if config.pretrain:
            assert pathlib.Path(config.pretrain).exists()
            # logger.info('Finetune with: {}'.format(config.pretrain))
            model.load_from_checkpoint(config.pretrain, config=config, map_location='cpu')
            resume_ckpt = None
        else:
            resume_ckpt = None
    if os.path.exists(resume_ckpt):
        checkpoint = torch.load('epoch=38-step=935.ckpt')
        model.load_state_dict(checkpoint)
    return model


def predict_im(img):
    config = json.load(open('pretrain.json'))
    #with_gpu = False


    config = easydict.EasyDict(config)
    # model = FOTSModel.load_from_checkpoint(checkpoint_path=model_path,
    #                                        map_location='cpu', config=config)
    model = load_model(config,True)
    model = model.to('cuda:0')
    model.eval()
    # for image_fn in input_dir.glob('*.jpg'):
        # try:
    # img = cv2.imread(str(image_fn))
    with torch.no_grad():
        # print(image_fn)
        ploy, im, pred = Toolbox.predict(img, model, False, '', with_gpu=True)
        # print(len(ploy))
        # except Exception as e:
        #     traceback.print_exc()
    return ploy, im, pred

# if __name__ == '__main__':
#     logger = logging.getLogger()

#     parser = argparse.ArgumentParser()
#     parser.add_argument('-o', '--output_dir', default='F:/project_2/New_folder/output', type=pathlib.Path,
#                         help='output dir for drawn images')
#     parser.add_argument('-i', '--input_dir', default='F:/project_2/New_folder/test/', type=pathlib.Path, required=False,
#                         help='dir for input images')
#     parser.add_argument('-c', '--config', default='pretrain.json', type=str,
#                         help='config file path (default: None)')
#     args = parser.parse_args('')
#     main(args)









