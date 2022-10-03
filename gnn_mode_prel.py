

from pickletools import optimize

import cv2
from myapp.FOTS.FOTS.utils.bbox import Toolbox
import torch
import numpy as np
from PIL import Image
import streamlit as st
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import os
import glob
import uuid
import base64
from io import BytesIO
import dgl
from tqdm import tqdm
import numpy as np
from PIL import Image
import scipy
try:
    import myapp.configs as cf
except:
    import configs as cf
import string
from myapp.HMM_implement import list_cha, root
try:
    from myapp.FOTS.eval import predict_im
    # from myapp.FOTS.FOTS.util import keys
    from myapp.kie.backend.models import load_text_detect, load_text_recognize, load_saliency
    from myapp.kie.backend.backend_utils import (
        NpEncoder,
        run_ocr,
        make_warp_img,
        resize_and_pad,
        get_group_text_line,
    )
    from myapp.kie.backend.kie.kie_utils import (
        load_gate_gcn_net,
        prepare_pipeline,
    )
except:
    from FOTS.eval import predict_im
    from kie.backend.models import load_text_detect, load_text_recognize, load_saliency
    from kie.backend.backend_utils import (
        NpEncoder,
        run_ocr,
        make_warp_img,
        resize_and_pad,
        get_group_text_line,
    )
    from kie.backend.kie.kie_utils import (
        load_gate_gcn_net,
        prepare_pipeline,
    )

keys = string.printable+'ĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐÐÊÉẾÈỀẺỂẼỄẸỆÍÌỈĨỊÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢƯÚỨÙỪỦỬŨỮỤỰÝỲỶỸỴăâáắấàằầảẳẩãẵẫạặậđêéếèềẻểẽễẹệíìỉĩịôơóốớòồờỏổởõỗỡọộợưúứùừủửũữụựýỳỷỹỵ'

# class GreedyCTCDecoder(torch.nn.Module):
#     def __init__(self, labels, tree,blank=0):
#         super().__init__()
#         self.labels = labels
#         self.blank = blank
#         self.tree = tree

#     def forward(self, emission: torch.Tensor,to_string=True):
#         """Given a sequence emission over labels, get the best path
#         Args:
#           emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

#         Returns:
#           List[str]: The resulting transcript
#         """
#         oute =''
#         index = torch.argmax(emission, dim=-1)
#         for inde in emission:
#             list_num = list_cha(oute,self.tree,self.labels)
#             inde = inde#*0.6+ list_num*0.4
#             inde = torch.argmax(inde, dim=-1)  # [num_seq,]
#             # inde = torch.unique_consecutive(inde, dim=-1)
#             # predict(oute,'',self.tree)
#             if inde != self.blank:
#                 oute += self.labels[inde]
#             else:
#                 oute +='-'
#         last = ''
#         final = ''
#         for st in oute:
#             if st!= last:
#                 final += st
#             last = st
#         final = final.strip('-')
#         if to_string:
#             return oute.strip('-'), index
#         else:
#             return index

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



list_sample = glob.glob('F:/project_2/opencv-haar-classifier-training/samples/*.txt')
def load_model():
    gcn_net = load_gate_gcn_net(cf.device, cf.kie_weight_path)
    detector = predict_im
    return gcn_net, detector

device= cf.device
#################################################################
#util function
################################################################3




def prepare_data(boxes,pre):
    origin_boxes = boxes #boxes = [8 position , w, h]
    node_nums = len(pre)  
    boxes = boxes
    src = []
    dst = []
    edge_data = []
    for i in range(node_nums):
        for j in range(node_nums):
            if i == j:
                continue

            edata = []
            # y distance
            y_distance = np.mean(boxes[i][:8][1::2]) - np.mean(boxes[j][:8][1::2])
            # w = boxes[i, 8]
            h = boxes[i][9]
            if np.abs(y_distance) > 3 * h:
                continue

            x_distance = np.mean(boxes[i][:8][0::2]) - np.mean(boxes[j][:8][0::2])
            edata.append(y_distance)
            edata.append(x_distance)

            edge_data.append(edata)
            src.append(i)
            dst.append(j)

    edge_data = np.array(edge_data)
    g = dgl.DGLGraph()
    g.add_nodes(node_nums)
    g.add_edges(src, dst)
    leng_pre = [np.max(np.where(x>0))+1 for x in pre]
    boxes, edge_data, text, text_length = prepare_pipeline(
        np.array(boxes), edge_data, pre, leng_pre
    )
    boxes = torch.from_numpy(boxes).float()
    edge_data = torch.from_numpy(edge_data).float()

    tab_sizes_n = g.number_of_nodes()
    tab_snorm_n = torch.FloatTensor(tab_sizes_n, 1).fill_(1.0 / float(tab_sizes_n))
    snorm_n = tab_snorm_n.sqrt()

    tab_sizes_e = g.number_of_edges()
    tab_snorm_e = torch.FloatTensor(tab_sizes_e, 1).fill_(1.0 / float(tab_sizes_e))
    snorm_e = tab_snorm_e.sqrt()

    max_length = np.array(len(pre)).max()
    # new_text = text
    texts = text
    # texts = np.concatenate(new_text)

    texts = torch.stack(text)
    text_length = torch.from_numpy(np.array(text_length))

    graph_node_size = [g.number_of_nodes()]
    graph_edge_size = [g.number_of_edges()]

    return (
        g,
        boxes,
        edge_data,
        snorm_n,
        snorm_e,
        texts,
        text_length,
        origin_boxes,
        graph_node_size,
        graph_edge_size,
    )




def sort_boundingbox(boxes = None, polys = None, text_label= None):
    line_index =[0 for i in range(len(text_label))]
    line_index[0] =1
    # decoder_ctc =  GreedyCTCDecoder(keys,blank= 236)
    keyss = keys + '   '
    # numerical_order = list(range(len(texts_concat_bboxs[0])))
    if boxes is not None:
        texts_concat_bboxs =  [boxes , text_label]
        numerical_order = list(range(len(texts_concat_bboxs[0])))
        len_ = len(numerical_order)-1
        for i in range(len_):
            for j in range(len_):
                if texts_concat_bboxs[0][j][1]>texts_concat_bboxs[0][j+1][1]:
                    temp = numerical_order[j+1]
                    numerical_order[j+1] = numerical_order[j]
                    numerical_order[j] = temp
        
        for j in range(len(texts_concat_bboxs[0])-1):
            if texts_concat_bboxs[0][j+1][1]-5<texts_concat_bboxs[0][j][7]:
                line_index[j+1] = line_index[j]
            else:
                line_index[j+1] = line_index[j]+1
            # a= decoder_ctc.forward(torch.tensor(texts_concat_bboxs[1][i][0]),to_string=True)
        max_line = max(line_index)
        for line_num in range(1,max_line+1):
            index_same_l = [id for id,k in enumerate(line_index) if k== line_num]
            if len(index_same_l)>2:
                for i in index_same_l:
                    for j in index_same_l[:-1]:
                        if is_left_of_word(texts_concat_bboxs[0][numerical_order[j]],texts_concat_bboxs[0][numerical_order[j+1]], follow_height= True):
                        # if texts_concat_bboxs[0][numerical_order[j]][0,0]>texts_concat_bboxs[0][numerical_order[j+1]][0,0] and (line_index[j]== line_index[j+1]):
                            temp = numerical_order[j+1]
                            numerical_order[j+1] = numerical_order[j]
                            numerical_order[j] = temp
        texts_concat_bboxs[0] = texts_concat_bboxs[0][numerical_order]
        label_result = [[texts_concat_bboxs[1][i][0],texts_concat_bboxs[1][i][1]] for i in numerical_order]
        # label_result = texts_concat_bboxs[1][0][numerical_order]
        return texts_concat_bboxs[0],label_result,line_index,numerical_order
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
            if len(texts_concat_bboxs[1][numerical_order[j]][0].shape)==1:
                a= [keyss[inde-1] for inde in texts_concat_bboxs[1][numerical_order[j]][0]]
                print(''.join(a).strip(' '))
        max_line = max(line_index)
        for line_num in range(1,max_line+1):
            index_same_l = [id for id,k in enumerate(line_index) if k== line_num]
            if len(index_same_l)>2:
                for i in index_same_l:
                    for j in index_same_l[:-1]:
                        if is_left_of_word(texts_concat_bboxs[0][numerical_order[j]],texts_concat_bboxs[0][numerical_order[j+1]],follow_height= True):
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
        center1 = [sum(box1[:-2][0::2])/4,sum(box1[:-2][1::2])/4]
        center2 = [sum(box2[:-2][0::2])/4,sum(box2[:-2][1::2])/4]
        w1= max(box1[:-2][0::2])-min(box1[:-2][0::2])
        if center1[0]>center2[0] and center1[1]-w1*0.25<center2[1] and center1[1]+w1*0.25>center2[1]:
            return True
        else:
            return False
    else:
        center1 = [sum(box1[:,0])/4,sum(box1[:,1])/4]
        center2 = [sum(box2[:,0])/4,sum(box2[:,1])/4]
        w1= max(box1[:,0])-min(box1[:,0])
        if center1[0]>center2[0] and center1[1]-w1*0.25<center2[1] and center1[1]+w1*0.25>center2[1]:
            return True
        else:
            return False
#############################################################

gcn_net,detector= load_model()

def extract_each(score,text):
    type =[[] for i in range(7)]
    title_word =[['ID'],['ho','va','ten','name'],['ngay', 'sinh', 'day','of','birth'],['giới','tính','sex'],['quốc','tịch','nationality'],['quê','quán','place','of','origin'],['nơi','thường','trú','place','of','residence'] ]
    for i in range(score.shape[0]):
        type_str = np.argmax(score[i])
        if type_str >1:
            if text[i] in title_word[type_str]:
                continue
            else:
                type[type_str].append(text)
    return type
# training model
def gcn_pre(img):
    h,w = img.shape[:2]
    ploy, imge, pred = detector(img)
    ploy = np.array(ploy)
    ploy =ploy.reshape(ploy.shape[0],-1)
    ploy =np.concatenate((ploy,np.array([h,w]*ploy.shape[0]).reshape(-1,2)),1)
    # simple training with batch_size = 1\
    pred = [pred[0].cpu().numpy(),pred[1].cpu().numpy()]
    decoder_ctc =  GreedyCTCDecoder(keys+'  ',root)
    texts=[]
    polys = []
    booxes = ploy
    w,h = imge.shape[:2]
    ratio_w = w / 640
    ratio_h = h / 640
    if (booxes is not None) and (len(booxes)>0):
        boxes = booxes[:, :8].reshape((-1, 4, 2))
        boxes= boxes.astype(float)

        # boxes[:, :, 0] *= ratio_w
        # boxes[:, :, 1] *= ratio_h
        boxes= boxes.astype(int)
        for box in boxes:
            box = Toolbox.sort_poly(box.astype(np.int32))
           
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                print('wrong direction')
                continue
            poly = np.array([[box[0, 0], box[0, 1]], [box[1, 0], box[1, 1]], [box[2, 0], box[2, 1]],
                            [box[3, 0], box[3, 1]]])
            polys.append(poly)
            p_area = Toolbox.polygon_area(poly)
            if p_area > 0:
                poly = poly[(0, 3, 2, 1), :]

            # if with_img:
            # cv2.rectangle(img,[label_box[0,0],label_box[0,1]],[label_box[2,0],label_box[2,1]],(0, 0, 255), thickness= 3, lineType=cv2.LINE_8)

            # cv2.rectangle(imge,(label_box[0,0],label_box[0,1]),(label_box[2,0],label_box[2,1]),(0, 0, 255), thickness= 3, lineType=cv2.LINE_8)

            cv2.rectangle(imge,(box[0,0],box[0,1]),(box[2,0],box[2,1]),(0, 255, 0), thickness= 3, lineType=cv2.LINE_8) 
    pred_ = [[pred[0][:,id],pred[1][id]] for id in range(len(pred[1]))]
    ploy,transcripss,line_index,numerical_order = sort_boundingbox(boxes = ploy, polys = None, text_label= pred_)
    string_text = ''
    content_line = ['' for i in range(max(line_index)+1)]
    text_str = []
    for id in range(pred[0].shape[1]):
        str_text,index = decoder_ctc.forward(torch.tensor(pred[0][:,id,:]),to_string=True)
        content_line[line_index[id]] += ' '+str_text.strip(' ')
        text_str.append(str_text)
        texts.append(index)
    string_text = '\n'.join(content_line)
    print('\n'.join(content_line))
    (
    batch_graphs,
    batch_x,
    batch_e,
    batch_snorm_n,
    batch_snorm_e,
    text,
    text_length,
    boxes,
    graph_node_size,
    graph_edge_size,
    )=prepare_data(ploy, texts)
    # support = support.to(device)
    batch_graphs = batch_graphs.to(device)
    batch_x = batch_x.to(device)
    batch_e = batch_e.to(device)

    text = text.to(device)
    text_length = text_length.to(device)
    batch_snorm_e = batch_snorm_e.to(device)
    batch_snorm_n = batch_snorm_n.to(device)

    batch_graphs = batch_graphs.to(device)
    batch_scores = gcn_net.forward(
        batch_graphs,
        batch_x,
        batch_e,
        text,
        text_length,
        batch_snorm_n,
        batch_snorm_e,
        graph_node_size,
        graph_edge_size,
    )
    type = extract_each(batch_scores.detach().numpy(),text_str)
    return ploy,pred,batch_scores,imge, content_line, type
if __name__ == "__main__":
    img= cv2.imread("F:/project_2/New_folder/data/downloads/094343_b.jpg")
    ploy,pred,batch_scores,imge, string_text = gcn_pre(img)
    # img = np.array(img).transpose(-1,0,1)
    cv2.imshow("img",img)
    cv2.waitKey()
    print(ploy)
    
