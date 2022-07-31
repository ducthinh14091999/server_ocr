
import copy
import os
from pickletools import optimize
import time
import random
import cv2
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

try:
    from myapp.FOTS.eval import predict_im
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
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor,to_string=True):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        if to_string:
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank]
            joined = "".join([self.labels[i] for i in indices])
            return joined.replace("|", " ").strip().split()
        else:
            return indices



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
    leng_pre = [np.max(np.where(x>0)) for x in pre]
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


#############################################################

gcn_net,detector= load_model()

# training model
def gcn_pre(img):
    h,w = img.shape[:2]
    ploy, imge, pred = detector(img)
    ploy = np.array(ploy)
    ploy =ploy.reshape(ploy.shape[0],-1)
    ploy =np.concatenate((ploy,np.array([h,w]*ploy.shape[0]).reshape(-1,2)),1)
    # simple training with batch_size = 1\
    pred = [pred[0].cpu().numpy(),pred[1].cpu().numpy()]
    decoder_ctc =  GreedyCTCDecoder(string.printable+'  ')
    texts=[]
    for id in range(pred[0].shape[1]):
        text = decoder_ctc.forward(torch.tensor(pred[0][:,id,:]),to_string=False)
        texts.append(text)
    text = texts
    string_text = ''
    for id in range(pred[0].shape[1]):
        str_text = decoder_ctc.forward(torch.tensor(pred[0][:,id,:]),to_string=True)[0]
        string_text += str_text
    print(string_text)
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
    )=prepare_data(ploy, text)
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
    return ploy,pred,batch_scores,imge, string_text
if __name__ == "__main__":
    img= cv2.imread("F:/project_2/New_folder/data/downloads/094343_b.jpg")
    ploy,pred,batch_scores,imge, string_text = gcn_pre(img)
    # img = np.array(img).transpose(-1,0,1)
    cv2.imshow("img",img)
    cv2.waitKey()
    print(ploy)
    
