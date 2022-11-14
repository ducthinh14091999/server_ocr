from sklearn.neighbors import KNeighborsClassifier
import glob
import cv2
import numpy as np
from myapp.FOTS.FOTS.utils.util import keys


list_sample = glob.glob('F:/project_2/opencv-haar-classifier-training/samples/*.txt')
def knn_classify():
    dataset = dataloader_gnn()
    dataset =  posision_relative(dataset)
    boxes = []
    label = []
    for data in dataset:
        for id in range(len(data['box'])):
            boxes.append(data['box'][id][:8])
            label.append(int(data['class'][id]))

    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(boxes,label)
    return neigh

def posision_relative(dataset):
    for data in dataset:
        name = data['name']
        img = cv2.imread(name)
        h,w = img.shape[:2]
        boxes = np.array(data['box'])
        boxes[:,0::2]/= (w/3)
        boxes[:,1::2]/= h
        data['box'] = boxes
    return dataset


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
            dataset.append({'name':'F:/project_2/KIE_invoice_minimal/results/crop/cluster0/'+file.split('\\')[-1].split('.')[0]+'.jpg','box':boxess, 'text':text,'class': classes,'text_length':texts_length})
    return dataset

def make_text_encode(text):
    text_encode = []
    for t in text.upper():
        if t not in keys:
            text_encode.append(keys.index(" "))
        else:
            text_encode.append(keys.index(t))
    return np.array(text_encode)