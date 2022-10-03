
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
import matplotlib.pyplot as plt
import networkx as nx
from fastapi import FastAPI, File, UploadFile
import scipy
from backend.models import load_text_detect, load_text_recognize, load_saliency
from backend.backend_utils import (
    NpEncoder,
    run_ocr,
    make_warp_img,
    resize_and_pad,
    get_group_text_line,
)
from backend.text_detect.config import craft_config
from backend.saliency.infer import run_saliency
import configs as cf
from backend.kie.kie_utils import (
    load_gate_gcn_net,
    run_predict,
    vis_kie_pred,
    postprocess_scores,
    prepare_pipeline,
    make_text_encode,
    prepare_graph
)

list_sample = glob.glob('F:/project_2/opencv-haar-classifier-training/samples/*.txt')
def load_model():
    gcn_net = load_gate_gcn_net(cf.device, cf.kie_weight_path)
    config = Cfg.load_config_from_name("vgg_seq2seq")
    config["cnn"]["pretrained"] = False
    config["device"] = cf.device
    config["predictor"]["beamsearch"] = False
    detector = Predictor(config)

    return gcn_net, detector

device= cf.device
#################################################################
#util function
################################################################3


class OutputHook:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []
hook = OutputHook()
hook_handles = []


def dataloader_gnn():
    dataset =[]
    
    for file in list_sample:
        boxess=[]
        text=[]
        classes=[]
        texts_length =[]
        with open(file,'r', encoding='utf-8') as lines:
            for line in lines:
                
                line = line.strip('\n').split('*')
                boxes= np.zeros(10)
                boxes[:8] = np.array(line[1:-1]).astype(int)
                boxes[8:] = [np.max(boxes[:8][0::2]) - np.min(boxes[:8][0::2]),np.max(boxes[:8][1::2]) - np.min(boxes[:8][1::2])]
                recognition =make_text_encode(line[0])
                text_length = len(line[0])
                class_gnn  = line[-1]
                boxess.append(boxes)
                text.append(recognition)
                classes.append(class_gnn)
                texts_length.append(text_length)
            dataset.append({'box':boxess, 'text':text,'class': classes,'text_length':texts_length})
    return dataset


def prepare_train(dataset,index):
    origin_boxes = dataset[index]['box'].copy() #boxes = [8 position , w, h]
    node_nums = len(dataset[index]['text_length'])  
    boxes = dataset[index]['box']
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

    boxes, edge_data, text, text_length = prepare_pipeline(
        np.array(boxes), edge_data, dataset[index]['text'], dataset[index]['text_length']
    )
    boxes = torch.from_numpy(boxes).float()
    edge_data = torch.from_numpy(edge_data).float()

    tab_sizes_n = g.number_of_nodes()
    tab_snorm_n = torch.FloatTensor(tab_sizes_n, 1).fill_(1.0 / float(tab_sizes_n))
    snorm_n = tab_snorm_n.sqrt()

    tab_sizes_e = g.number_of_edges()
    tab_snorm_e = torch.FloatTensor(tab_sizes_e, 1).fill_(1.0 / float(tab_sizes_e))
    snorm_e = tab_snorm_e.sqrt()

    max_length = np.array(dataset[index]['text_length']).max()
    new_text = [
        np.expand_dims(np.pad(t, (0, max_length - t.shape[0]), "constant"), axis=0)
        for t in text
    ]
    texts = np.concatenate(new_text)

    texts = torch.from_numpy(np.array(texts))
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

def one_hot_label(train_ids):
    label=dataset[train_ids]['class']
    encoded = []
    for sample in label:
        oh_vector = np.zeros(9)
        if int(sample)==-1:
            oh_vector[0] = 1
        else:   
            oh_vector[int(sample)+1] = 1
        encoded.append(oh_vector)
    return np.array(encoded)
#############################################################
dataset = dataloader_gnn()
train_ids = list(range(len(dataset)))
gcn_net, detector = load_model()
optimizer = torch.optim.AdamW(
    gcn_net.parameters(), lr=0.0001, weight_decay=0.9)

# optimizer =torch.optim.SGD(gcn_net.parameters(), lr=0.001)  
# training model
for layer in gcn_net.modules():
    if isinstance(layer, torch.nn.Linear):
        handle = layer.register_backward_hook(hook)
        hook_handles.append(handle)
for epoch in range(2000):
    gcn_net.train()
    random.shuffle(train_ids)
    t1 = time.time()
    train_losses = 0
    train_accs = []

    batch_losses = []
    train_ids= train_ids[:10]
    # simple training with batch_size = 1
    for img_index, img_id in tqdm(enumerate(train_ids)):
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
        )=prepare_train(dataset,img_index)
        # G = dgl.to_networkx(batch_graphs)
        # options = {
        #     'node_color': 'black',
        #     'node_size': 50,
        #     'width': 0.5,
        # }
        # plt.figure(figsize=[15,7])
        # nx.draw(G, **options)
        # nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
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
        label = one_hot_label(img_index)
        loss = gcn_net.loss(batch_scores,torch.tensor(label))

        train_losses += loss.item()
        batch_losses.append(loss.item())
        if img_index % 100 == 0:
            print("train loss: {:.5f} ".format(np.mean(batch_losses)))
            batch_losses = []


        # train_accs.append(acc.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(gcn_net.parameters(), 100)
        # for id,param in enumerate(optimizer.param_groups[0]['params']):
        #     if param.grad is not None:
        #         print('layer ',id,param.grad.shape,param.grad.mean() )
                # if len(param.grad.shape)>1:
                #     plt.imshow(param.grad)
                #     plt.show()
        # plt.imshow(np.ones((9,9)))
        # plt.show()
    train_losses /= (img_index + 1)
    # acc = np.mean(train_accs)
    t2 = time.time()
    print(
        "Epoch:",
        "%04d" % (epoch + 1),
        "time: {:.5f}, loss: {:.5f},".format(
            (t2 - t1), train_losses
        ),
    )
    torch.save(gcn_net.state_dict(), 'GCN_model.pkl')