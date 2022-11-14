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




def main(config, resume: bool):
    loss_model = FOTSLoss(config)
    
    thres = [0.45,0.5,0.6,0.65,0.7,0.85,0.88,0.89,0.9]
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
    is_training = True
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
            # else:
            boxes_non_text = outputs['bboxes']
            booxes = outputs['bboxes']
            if (booxes is not None) and (len(booxes)>0):
                boxes = booxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] *= ratio_w
                boxes[:, :, 1] *= ratio_h

                boxes_non_text = boxes_non_text[:, :8].reshape((-1, 4, 2))
                boxes_non_text[:,:,0]*= ratio_w
                boxes_non_text[:,:,1]*= ratio_h

                label_boxes = outputs['label_boxes']
                label_boxes[:, :, 0] *= ratio_w
                label_boxes[:, :, 1] *= ratio_h
                label_b = [[0 for i in range(len(label_boxes))]for j in range(len(thres))]
                boxx = [[0 for i in range(len(boxes))] for j in range(len(thres))]
                box_non = [[0 for i in range(len(boxes_non_text))] for j in range(len(thres))]


                for id_p,box in enumerate(boxes_non_text):
                    max_non_iou = 0
                    for id_l,label_box in enumerate(label_boxes):
                        if torch.is_tensor(box):
                            box = Toolbox.sort_poly(box.cpu().numpy().astype(np.int32))
                        else:
                            box = Toolbox.sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                            print('wrong direction')
                            continue
                        max_non_iou = max_non_iou if max_non_iou>intersection(label_box, box) else intersection(label_box, box)
                        
                        
                        for id_t,thes in enumerate(thres):
                            if max_non_iou>thes:
                                box_non[id_t][id_p]+=1
                    for id_t,thes in enumerate(thres):
                        FN[id_t] += len(list(filter(lambda x: x >0, box_non[id_t])))


                for id_p,box in enumerate(boxes):
                    max_iou =0
                    max_non_iou = 0
                    for id_l,label_box in enumerate(label_boxes):
                        if torch.is_tensor(box):
                            box = Toolbox.sort_poly(box.cpu().numpy().astype(np.int32))
                        else:
                            box = Toolbox.sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                            print('wrong direction')
                            continue
                        max_iou = max_iou if max_iou>intersection(label_box, box) else intersection(label_box, box)
                        max_non_iou = max_non_iou if max_non_iou>intersection(label_box, box) else intersection(label_box, box)
                        
                        
                        for id_t,thes in enumerate(thres):

                            if max_iou >thes:
                                label_b[id_t][id_l]+=1
                                boxx[id_t][id_p]+=1
                    for id_t,thes in enumerate(thres):
                        TP[id_t] += len(list(filter(lambda x: x >0, label_b[id_t])))
                        # FN[id_t] += len(list(filter(lambda x: x ==0, label_b[id_t])))
                        FP[id_t] += len(list(filter(lambda x: x==0, boxx[id_t])))
                    ious.append(max_iou)
                # elif(len(boxes) >= len(label_boxes)):
                #     print('have False positive')
                #     FP += len(boxes) - len(label_boxes)
                # else:
                #     print('have False nagative')
                #     FN = len(label_boxes)- len(boxes)
            # except:
            #     print('error')
    # for precent in range(40,100):
    #     result =[]
    #     for iou in ious:
    #         if iou>precent/100:
    #             result.append(True)
    #         else:
    #             result.append(False)
    for id_t,thes in enumerate(thres):
        print('thres is: ',thes)
        print(TP[id_t],FP[id_t],FN[id_t])
        print(TP[id_t]/(TP[id_t]+FP[id_t]),TP[id_t]/(TP[id_t]+FN[id_t]))
        PEs.append(TP[id_t]/(TP[id_t]+FP[id_t])) 
        RCs.append(TP[id_t]/(TP[id_t]+FN[id_t]))
    plt.plot(PEs,RCs)
    plt.show()

                


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