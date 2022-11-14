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
from torch.hub import load_state_dict_from_url
from torch.optim.optimizer import Optimizer
from torch.utils.data import dataloader
from torch.optim.lr_scheduler import  ReduceLROnPlateau
from FOTS.FOTS.model.model import FOTSModel
from FOTS.FOTS.model.loss import *
# import wandb
import cv2
from FOTS.FOTS.utils.util import keys
from FOTS.FOTS.utils.bbox import Toolbox

from FOTS.FOTS.data_loader.data_module import SynthTextDataModule, ICDARDataModule
from pytorch_lightning.callbacks import Callback
import torch.optim as optim
cross_valid = True
data_train = True
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
        print(final)
        if to_string:
            return final
        else:
            return index


def sort_boundingbox(boxes = None, polys = None, text_label= None):
    line_index =[0 for i in range(len(text_label))]
    line_index[0] =1
    decoder_ctc =  GreedyCTCDecoder(keys,blank= 236)
    keyss = keys + '   '
    # numerical_order = list(range(len(texts_concat_bboxs[0])))
    if boxes is not None:
        texts_concat_bboxs =  [boxes , text_label]
        numerical_order = list(range(len(texts_concat_bboxs[0])))
        for i in enumerate(numerical_order):
            for j in enumerate(numerical_order):
                if texts_concat_bboxs[0][j][1]>texts_concat_bboxs[0][j+1][1]:
                    temp = numerical_order[j+1]
                    numerical_order[j+1] = numerical_order[j]
                    numerical_order[j] = temp
        
        for j in enumerate(range(len(texts_concat_bboxs[0])-1)):
            if texts_concat_bboxs[0][j+1][1]-5<texts_concat_bboxs[0][j][7]:
                line_index[j+1] = line_index[j]
            else:
                line_index[j+1] = line_index[j]+1
            # a= decoder_ctc.forward(torch.tensor(texts_concat_bboxs[1][i][0]),to_string=True)
        return texts_concat_bboxs[1],texts_concat_bboxs[0],line_index
    elif polys is not None:
        
        texts_concat_bboxs =  [polys , text_label]
        numerical_order = list(range(len(texts_concat_bboxs[0])))
        len_ = len(numerical_order)-1
        for i in range(len_):
            for j in range(len_):
                if (texts_concat_bboxs[0][numerical_order[j]][0,1]>texts_concat_bboxs[0][numerical_order[j+1]][0,1]) and \
                texts_concat_bboxs[0][numerical_order[j]][3,1]>texts_concat_bboxs[0][numerical_order[j+1]][3,1]:
                    temp = numerical_order[j+1]
                    numerical_order[j+1] = numerical_order[j]
                    numerical_order[j] = temp
        


        
        for j in range(len_):
            if texts_concat_bboxs[0][numerical_order[j+1]][0,1]<texts_concat_bboxs[0][numerical_order[j]][3,1]:
                line_index[j+1] = line_index[j]
            else:
                line_index[j+1] = line_index[j]+1
            # if len(texts_concat_bboxs[1][numerical_order[j]][0].shape)==1:
            #     a= [keyss[inde-1] for inde in texts_concat_bboxs[1][numerical_order[j]][0]]
            #     print(''.join(a).strip(' '))
        max_line = max(line_index)
        for line_num in range(1,max_line+1):
            index_same_l = [id for id,k in enumerate(line_index) if k== line_num]
            if len(index_same_l)>2:
                for i in index_same_l:
                    for j in index_same_l[:-1]:
                        if is_left_of_word(texts_concat_bboxs[0][numerical_order[j]],texts_concat_bboxs[0][numerical_order[j+1]]):
                        # if texts_concat_bboxs[0][numerical_order[j]][0,0]>texts_concat_bboxs[0][numerical_order[j+1]][0,0] and (line_index[j]== line_index[j+1]):
                            temp = numerical_order[j+1]
                            numerical_order[j+1] = numerical_order[j]
                            numerical_order[j] = temp
        texts_concat_bboxs[0] = texts_concat_bboxs[0][numerical_order]
        label_result = [[texts_concat_bboxs[1][i][0],texts_concat_bboxs[1][i][1]] for i in numerical_order]
        # label_result = texts_concat_bboxs[1][0][numerical_order]
        return texts_concat_bboxs[0],label_result,line_index,numerical_order
    else:
        return boxes, polys, text_label

def is_left_of_word(box1,box2,follow_height= False):
    
    # w2,h2 = max(box2[:,0])-min(box2[:,0]), max(box2[:,1])-min(box2[:,1])
    if follow_height:
        min1,max1 = min(box1[:,1]),max(box1[:,1])
        min2,max2 = min(box2[:,1]),max(box2[:,1])
        h2 = max(box2[:,1])-min(box2[:,1])
        # if 
    else:
        center1 = [sum(box1[:,0])/4,sum(box1[:,1])/4]
        center2 = [sum(box2[:,0])/4,sum(box2[:,1])/4]
        w1= max(box1[:,0])-min(box1[:,0])
        if center1[0]>center2[0] and center1[1]-w1*0.25<center2[1] and center1[1]+w1*0.25>center2[1]:
            return True
        else:
            return False



def main(config, resume: bool):
    loss_model = FOTSLoss(config)
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
    decoder_ctc =  GreedyCTCDecoder(keys,blank= 236)
    
    is_training = True
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        if is_training:
          data_input = data_module.train_dataloader()
        else:
          data_input = data_module.val_dataloader()
        for i, data in enumerate(iter(data_input)):
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
            # transcrips[0] = [transcrip for idd,transcrip in enumerate(transcrips[0]) if idd not in remove_roi_id]
            if  not is_training:
              transcrips = list(transcrips)
            # print(transcrips)
            if config.data_loader.dataset == 'synth800k':
                # transcrips[1]=transcrips[1][outputs["indices"]]
                # transcrips[0]=transcrips[0][outputs["indices"]]
                transcrips = [transcrips[0][outputs["indices"]],transcrips[1][outputs["indices"]]]
            # else:
            #     transcrips[1]=transcrips[1][outputs["indices"]]
            #     transcrips[0]=transcrips[0][outputs["indices"]]
            #     pass
            loss_dict = loss_model(score_map,outputs['score_maps'],geo_map,outputs['geo_maps'],transcrips,outputs['transcripts'],training_marks)
            loss = loss_dict['reg_loss'] + loss_dict['cls_loss'] + loss_dict['recog_loss']
            polys = []
            label_polys =[]
            # outputs['bboxes'],transcripss,line_index= sort_boundingbox(boxes = None, polys = outputs['bboxes'].cpu().numpy(), text_label= [[outputs['transcripts'][0][i],outputs['transcripts'][1][i]] for i in range(outputs['transcripts'][0].shape[1]) ])
            transcr = [[transcrips[0][i],transcrips[1][i]] for i in range(len(transcrips[1]))] 
            outputs['bboxes'],transcripss,line_index,numerical_order= sort_boundingbox(boxes = None, polys = outputs['bboxes'].cpu().numpy(), text_label=transcr )
            
            if data_train:
                booxes = bboxes
            else:
                
                booxes = outputs['bboxes']
            if (booxes is not None) and (len(booxes)>0):
                boxes = booxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] *= ratio_w
                boxes[:, :, 1] *= ratio_h
                label_boxes = outputs['label_boxes'].cpu().numpy()  
                label_boxes[:, :, 0] *= ratio_w
                label_boxes[:, :, 1] *= ratio_h
                for box,label_box in zip(boxes,label_boxes):
                    box = Toolbox.sort_poly(box.cpu().numpy().astype(np.int32))
                    label_box = Toolbox.sort_poly(label_box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        print('wrong direction')
                        continue
                    poly = np.array([[box[0, 0], box[0, 1]], [box[1, 0], box[1, 1]], [box[2, 0], box[2, 1]],
                                    [box[3, 0], box[3, 1]]])
                    polys.append(poly)
                    label_polys.append(label_box.reshape(4,2))
                    p_area = Toolbox.polygon_area(poly)
                    if p_area > 0:
                        poly = poly[(0, 3, 2, 1), :]

                    # if with_img:
                    # cv2.rectangle(img,[label_box[0,0],label_box[0,1]],[label_box[2,0],label_box[2,1]],(0, 0, 255), thickness= 3, lineType=cv2.LINE_8)

                    cv2.rectangle(img,(label_box[0,0],label_box[0,1]),(label_box[2,0],label_box[2,1]),(0, 0, 255), thickness= 3, lineType=cv2.LINE_8)

                    cv2.rectangle(img,(box[0,0],box[0,1]),(box[2,0],box[2,1]),(0, 255, 0), thickness= 3, lineType=cv2.LINE_8) 
                plt.imshow(img)
                plt.show()

            
            string_text = ''
            content_line = ['' for i in range(max(line_index)+1)]
            for id in range(outputs['transcripts'][0].shape[1]):
                result_text_ = outputs['transcripts'][0][id] #nn.Softmax(dim=1)
                result_text = transcripss[id][0]
                # plt.imshow(result_text.data.cpu().numpy().transpose())
                # plt.show()
                str_text = decoder_ctc.forward(torch.tensor(result_text_),to_string=True)
                # result_text = (transcrips[0][id]) #nn.Softmax(dim=1)
                
                keyss = keys + '   '
                result = [keyss[inde-1] for inde in result_text]
                label_text =''.join(result)
                content_line[line_index[id]] += ' '+label_text.strip(' ')
                if np.min(polys[id]) >0 and (polys[id][2,1]-polys[id][0,1])>0 and (polys[id][2,0]-polys[id][0,0])>0:
                    plt.imshow(img[polys[id][0,1]:polys[id][2,1],polys[id][0,0]:polys[id][2,0],:])
                    plt.title('predict_text'+str_text)
                    print(intersection(label_boxes[id], boxes[id]))
                    # plt.rcParams["figure.figsize"] = (2,20)
                    plt.show()
                    plt.imshow(img[label_polys[id][0,1]:label_polys[id][2,1],label_polys[id][0,0]:label_polys[id][2,0],:])
                    plt.title('label'+' '+label_text)
                    plt.show()
                    string_text += str_text
                    string_text += ' '
            print(string_text)
            for i in content_line:
                print(i)
                


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