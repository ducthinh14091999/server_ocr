#CER, WER in ocr
import numpy as np
import fastwer
import matplotlib.pyplot as plt



import matplotlib.pyplot as plt
import argparse
import json
from loguru import logger
import os
import pathlib
import torch
from FOTS.FOTS.utils.lanms import *
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from easydict import EasyDict
import numpy as np
from FOTS.FOTS.model.model import FOTSModel
from FOTS.FOTS.model.loss import *
# import wandb
import cv2
from FOTS.FOTS.utils.util import keys
from FOTS.FOTS.utils.bbox import Toolbox

from FOTS.FOTS.data_loader.data_module import SynthTextDataModule, ICDARDataModule
from pytorch_lightning.callbacks import Callback
import torch.optim as optim
# from torchmetrics.detection.mean_ap import MeanAveragePrecision
cross_valid = True
data_train = False

from HMM_implement import list_cha, root

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
        # print([self.labels[inde] for inde in index])
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
            if st!= last and st!='0':
                final += st
            last = st
        final = final.strip(' ')
        # print(final)
        if to_string:
            return final, index
        else:
            return index



def main(config, resume: bool):
    loss_model = FOTSLoss(config)
    decoder_ctc =  GreedyCTCDecoder(keys+'  ',root)
    thres = [0,4,0.45,0.5,0.6,0.65,0.7,0.85,0.88,0.89,0.9]
    TP =[0]*len(thres)
    FP = [0]*len(thres)
    FN = [0]*len(thres)
    PEs =[]
    RCs =[]
    if not config.cuda:
        gpus = 0
        # device = torch.device('cuda:0')
    else:
        gpus = config.gpus
        # device = torch.device('cuda:0')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FOTSModel(config).to(device)
    # model=torch.load('epoch=38-step=935.ckpt', map_location="cuda:0")
    # model.load_from_checkpoint('epoch=38-step=935.ckpt')
    if resume and  pathlib.Path(config.pretrain).exists():
        assert pathlib.Path(config.pretrain).exists()
        resume_ckpt = config.pretrain
        logger.info('Resume training from: {}'.format(config.pretrain))
    else:
        if config.pretrain and  pathlib.Path(config.pretrain).exists() :
            # assert pathlib.Path(config.pretrain).exists()
            logger.info('Finetune with: {}'.format(config.pretrain))
            model.load_from_checkpoint(config.pretrain, config=config, map_location='cpu')
            resume_ckpt = None
        else:
            resume_ckpt = None
    # scheduler = ReduceLROnPlateau(mode='min')
    if config.data_loader.dataset == 'synth800k':
        data_module = SynthTextDataModule(config)
    else:
        data_module = ICDARDataModule(config)
    data_module.setup()


    root_dir = str(pathlib.Path(config.trainer.save_dir).absolute() / config.name)
    # print(root_dir)
    # checkpoint_callback = ModelCheckpoint(dirpath=root_dir + '/checkpoints', period=1)
    wandb_dir = pathlib.Path(root_dir) / 'wandb'
    if not wandb_dir.exists():
        wandb_dir.mkdir(parents=True, exist_ok=True)

    print(config)
    # wandb.config=config
    print(resume_ckpt)
    if resume_ckpt != None:
      if os.path.exists(resume_ckpt):
          checkpoint = torch.load(resume_ckpt)
          model.load_state_dict(checkpoint)
    
    ious=[]
    num_string = 0
    pestisive = 0
    is_training  = True
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        if is_training:
          data_input = data_module.train_dataloader()
        else:
          data_input = data_module.val_dataloader()
        for i, data in enumerate(iter(data_input)):
            # try:
            # get the inputs; data is a list of [inputs, labels]
            image_name,images,score_map,geo_map,training_marks,transcrips,bboxes,rois = data['image_names'], data['images'],data['score_maps'], data['geo_maps'],data['training_masks'], data['transcripts'], data['bboxes'], data['rois']
            images = images.to(device="cuda")
            score_map = score_map.to(device="cuda")
            geo_map = geo_map.to(device="cuda")
            bboxes = bboxes.to(device="cuda")
            # print(image_name)
            rois = rois.to(device="cuda")
            training_marks = training_marks.to(device="cuda")
            im_resized = cv2.resize(np.array(data['images'][0]).transpose([1,2,0]), dsize=(640, 640))
            # im_resized = (im_resized-127)/128
            h, w, _ = np.array(data['images'][0]).transpose([1,2,0]).shape
            ratio_w = w / 640
            ratio_h = h / 640
            img = cv2.imread(data['image_names'][0])
            img = cv2.resize(img,(640,640))
            # forward + backward + optimize
            outputs = model.forward(images,bboxes,rois)
            pred = outputs['transcripts']
            label_str = transcrips
            # transcrips[0] = [transcrip for idd,transcrip in enumerate(transcrips[0]) if idd not in remove_roi_id]
           
            # else:
            #     transcrips[1]=transcrips[1][outputs["indices"]]
            #     transcrips[0]=transcrips[0][outputs["indices"]]
            #     pass
            # loss_dict = loss_model(score_map,outputs['score_maps'],geo_map,outputs['geo_maps'],transcrips,outputs['transcripts'],training_marks)
            # loss = loss_dict['reg_loss'] + loss_dict['cls_loss'] + loss_dict['recog_loss']
            # print(loss_dict)

            
            # transcr = [[transcrips[0][i],transcrips[1][i]] for i in range(len(transcrips[1]))] 
            # pre_transcr = [[outputs['transcripts'][0][:,i],outputs['transcripts'][1][i]] for i in range(len(outputs['transcripts'][1]))]
            # outputs['bboxes'],pre_transcr,line_index_pre, numerical_order_pre= sort_boundingbox(boxes = None, polys = outputs['bboxes'].cpu().numpy(), text_label= pre_transcr)
            # outputs['label_boxes'],transcripss,line_index,numerical_order= sort_boundingbox(boxes = None, polys = outputs['label_boxes'].cpu().numpy(), text_label=transcr )
            
            # if data_train:
            #     booxes = bboxes[numerical_order]
            # else:]
            booxes = outputs['bboxes']
            if (booxes is not None) and (len(booxes)>0):
                boxes = booxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] *= ratio_w
                boxes[:, :, 1] *= ratio_h

                label_boxes = outputs['label_boxes']
                label_boxes[:, :, 0] *= ratio_w
                label_boxes[:, :, 1] *= ratio_h
                label_b = [0 for i in range(len(label_boxes))]
                boxx = [0 for i in range(len(boxes))] 
                posi_l = -1
                posi_b = -1
                for id_p,box in enumerate(boxes):
                    max_iou =0
                    posi_l = -1
                    posi_b = -1
                    for id_l,label_box in enumerate(label_boxes):
                        if torch.is_tensor(box):
                            box = Toolbox.sort_poly(box.cpu().numpy().astype(np.int32))
                        else:
                            box = Toolbox.sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                            print('wrong direction')
                            continue
                        if intersection(label_box, box)!= 0:
                            posi_l = id_l if max_iou<intersection(label_box, box) else posi_l
                            posi_b = id_p if max_iou<intersection(label_box, box) else posi_b
                            max_iou = max_iou if max_iou<intersection(label_box, box) else intersection(label_box, box)
                    if posi_l != -1:
                        str_text,index = decoder_ctc.forward(torch.tensor(pred[0][:,posi_b,:]),to_string=True)
                        # print(str_text)
                        img_debug = img.copy()
                        # cv2.rectangle(img_debug,(label_boxes[posi_l][0,0],label_boxes[posi_l][0,1]),(label_boxes[posi_l][2,0],label_boxes[posi_l][2,1]),(0, 0, 255), thickness= 3, lineType=cv2.LINE_8)
                        # cv2.rectangle(img_debug,(box[0,0],box[0,1]),(box[2,0],box[2,1]),(255, 0, 0), thickness= 3, lineType=cv2.LINE_8) 
                        # print(box, label_boxes[posi_l],max_iou)
                        # plt.imshow(img_debug)
                        # plt.show()       
                        result_text = transcrips[0][posi_l]
                        keyss = keys + '   '
                        result = [keyss[inde-1] for inde in result_text]
                        label_text =''.join(result).strip(' ')
                        print(str_text, label_text,fastwer.score_sent(str_text, label_text, char_level=True),max_iou)
                        pestisive += fastwer.score_sent(str_text, label_text, char_level=True)/100
                        num_string+=1
        print('this is average of wer:',pestisive/num_string)
        num_string = 0
        pestisive = 0
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='pretrain.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume',default=True, action='store_true',
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args('')

    config = None
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        # assert not os.path.exists(path), "Path {} already exists!".format(path)
    else:
        if args.resume is not None:
            logger.warning('Warning: --config overridden by --resume')
            config = torch.load(args.resume, map_location='cpu')['config']

    assert config is not None
    config = EasyDict(config)
    main(config, args.resume)
def CER(pred,label):
    matrix = np.zeros([len(pred),len(label)])
    N = len(label)
    D = 0
    S = 0
    for i in range(len(pred)):
        for  j in range(len(label)):
            if pred[i] != label[j]:
                D+=1
                S+=1
            else:
                matrix[i,j]+=1
    return  matrix
# print(CER('mitten','fitting'))
# print(CER('thinh','thinh'))
# print(fastwer.score_sent('mitten', 'fitting', char_level=True))
# plt.plot([0.6530768046790827,0.6381687910647875,0.6036061786872623,0.5168275688151353,0.23244007454533772], \
#          [0.9332418355855856,0.9254469313063063,0.9103586007882883,0.8294094876126126,0.2863967483108108])
# plt.show()
