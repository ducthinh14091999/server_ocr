import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM,Dense, Dropout, Activation, Flatten,BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Bidirectional
from tensorflow.keras.layers import Reshape, Lambda,Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
# from tensorflow.keras.layers.merge import add, concatenate
# from google.colab.patches import cv2_imshow

alphabet = " !"+'"'+"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"+'àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ\t_[]{}@\\<>'

# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
input_shape = (32, 800, 1)     # (128, 64, 1)

    # Make Network
inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 32, 800, 1)

    # Convolution layer (VGG)
inner = Conv2D(128, (3, 3), padding='same', name='conv1', activation='relu')(inputs)  # (None, 32,800,128)
inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,16, 400, 128)

inner = Conv2D(128, (3, 3), padding='same', name='conv2', activation='relu' )(inner)  # (None, 16, 400, 128)
inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 8, 200, 128)
inner = Conv2D(256, (3, 3), padding='same', name='conv2b', activation='relu')(inner)  # (None, 8, 200, 256)
inner = Conv2D(512, (3, 3), padding='same', name='conv3',activation='relu')(inner)  # (None, 8, 200, 512)
inner = MaxPooling2D(pool_size=(2, 1), name='max3')(inner)  # (None, 4,200, 512)
inner = Conv2D(512, (3, 3), padding='same', name='conv4',activation='relu')(inner)  # (None, 4, 200,512)


inner = Conv2D(1024, (3, 3), padding='same', name='conv5', activation='relu')(inner)  # (None, 4, 200, 512)
inner = MaxPooling2D(pool_size=(2, 1), name='max4')(inner)  # (None, 2, 200, 512)
#inner7b = BatchNormalization()(inner7)
inner = Conv2D(1024, (2, 2), padding='same',  activation='relu', name='con7')(inner)  # (None, 2, 200, 512)
inner = MaxPooling2D(pool_size=(2, 1), name='max5')(inner)


    # CNN to RNN
inner = squeezed = Lambda(lambda x: K.squeeze(x, 1))(inner)  # (None, 32, 2048)
#inner = Dense(256, activation='relu', name='dense1')(inner)  # (None, 32, 64)
    
lstm_1 = Bidirectional(LSTM(1024, return_sequences=True, go_backwards=True,name='lstm1'))(inner)
# lstm_2 = LSTM(1024, return_sequences=True, go_backwards=True, name='lstm2')(lstm_1)
#reversed_lstm_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_2b)

#lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])  # (None, 32, 1024)

    # transform1s RNN output to character activations:
y_pred = Dense(len(alphabet)+1,activation='softmax',name='dense2')(lstm_1) #(None, 32, 63)
labels = Input(name='the_labels', shape=[200], dtype='float32') # (None ,8)
input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)

model=Model(inputs=inputs,outputs=y_pred)
model.load_weights('word_overfit 1.h5')
print(model.summary())
def main():
    list_img=glob.glob('F:/de_cuong/project_2/New_folder/segment_line3/*.jpg')
    for address in list_img:

        img=cv2.imread(address,0)
        img=cv2.resize(img,(800,32))
        h,w=img.shape[:2]
        tam=np.zeros((32,800))
        if int(w*32//h)<800:
            img=cv2.resize(img,(int(w*32//h),32))
            tam[:,:int(w*32//h)]= img
        else:
            img=cv2.resize(img,(800,32))
            tam=img
        # tam=(tam-127)/256
        img=tam    

        img=np.array(img).reshape(1,img.shape[0],img.shape[1],1)
        # alphabet = " !"+'"'+"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"+'àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ'
        # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))
        # img=np.array(img).reshape(1,img.shape[0],img.shape[1],1)
        prediction=model.predict(img)
        out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                                greedy=False,)[0][0])
        i = 0
        int_to_char = dict((i, c) for i, c in enumerate(alphabet))
        for x in out:
            # print("original_text =  ", Y_train[i])
            print("predicted text = ", end = '')
            for p in x:  
                if int(p) != -1:
                    print(int_to_char[int(p)], end = '')       
            print('\n')
            i+=1
def predict_test(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=cv2.resize(img,(800,32))
    h,w=img.shape[:2]
    tam=np.zeros((32,800))
    if int(w*32//h)<800:
        img=cv2.resize(img,(int(w*32//h),32))
        tam[:,:int(w*32//h)]= img
    else:
        img=cv2.resize(img,(800,32))
        tam=img
    tam=(tam-127)/256
    img=tam    

    img=np.array(img).reshape(1,img.shape[0],img.shape[1],1)
    # alphabet = " !"+'"'+"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"+'àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # img=np.array(img).reshape(1,img.shape[0],img.shape[1],1)
    prediction=model.predict(img)
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                            greedy=False,)[0][0])
    i = 0
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    stri=''
    for x in out:
        # print("original_text =  ", Y_train[i])
        print("predicted text = ", end = '')
        for p in x:  
            if int(p) != -1:
                print(int_to_char[int(p)], end = '')
                stri+=int_to_char[int(p)]       
        print('\n')
        i+=1
    return stri
if __name__=='__main__':
    main()