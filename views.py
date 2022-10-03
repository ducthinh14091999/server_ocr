from django.shortcuts import render
from io import BytesIO
from .models import predict, result
from django.http import Http404, HttpResponse, HttpResponsePermanentRedirect
from django.shortcuts import render,redirect
from django.core.files.storage import FileSystemStorage
from .form import UploadFileForm
from .gnn_mode_prel import gcn_pre
from django.core.files.images import ImageFile
import cv2
from PIL import Image
import numpy as np
from matplotlib import cm
import base64
# from .test import predict_test
from .FOTS.eval import predict_im
# Create your views here.
def handle_uploaded_file(img):
    # image=Image.open(img).convert('RGB') 
    img.save('static/uploads/img.jpg')
    # cv2.imwrite('img.jpg',img)
def to_image(data):
    img = Image.fromarray(data, 'RGB')
    return img

def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "JPEG") # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8') 
def predict_view(request):
    if request.method == 'POST':
        name=request.FILES['img_input']
        form = UploadFileForm(request.POST, request.FILES)
        haar = False
        # print(request.POST)
        if form.is_valid():
            # if len(predict.objects.get(name=request.POST['name']))==1:
            #     Http404('name is exists')
            form.save()
            ID_card = True
            # height=predict.objects.get(name=request.POST['name']).img_input.height
            # width=predict.objects.get(name=request.POST['name']).img_input.width
            img=predict.objects.get(name=request.POST['name']).img_input
            img_object = predict.objects.get(name=request.POST['name'])
            img = Image.open(img).convert('RGB')
            img = np.array(img)
            if haar:
                img_orgin, boxes ,img_lists =haar_cascade_extact(img)
                if len(img_lists)>1:
                    img = img_lists[0]
            # print(img.shape)
            # print(height,width)
            # img=cv2.imread('uploads/img.jpg')
            # stri=predict_test(img)
            
            
            if (ID_card==True):
                ploy,pred,batch_scores,imge, string_text, type = gcn_pre(img)
                
            else:
                ploy, imge, pred = predict_im(img)
            
            print(imge.shape)
            cv2.imwrite('img_out.jpg',imge)
            output=result.objects
            output.all().delete()
            
            imgaa = Image.fromarray(imge,'RGB')
            image_uri = to_data_uri(imgaa)
            handle_uploaded_file(imgaa)
            imgaa = ImageFile(BytesIO(imgaa.tobytes()), name='img_out.jpg')
            predict.objects.all()[0].delete()
            cv2.imwrite('static/uploads/img1.jpg', imge)
            stri=string_text

            # print(imge.shape)
            print('all here')
            
            
            with open('result.txt','w',encoding='utf-8') as w:
                for st in stri:
                    w.write(st)
            return render(request,'myapp/result.html',{'img':img_object,"output":image_uri, 'string': string_text,'type':type})
    else:
        form = UploadFileForm()
    return render(request, 'myapp/predict.html', {'form' : form})
  
  
def success(request):
    return HttpResponse('successfully uploaded')
def main_page(request):
    if request.method == 'POST':
        pass
    else:
        pass
    return render(request, 'myapp/main.html')
def previous_result(request):
    stri = open('result.txt','r',encoding='utf-8').readline()

    return render(request,'myapp/last_result.html',{'stri':stri})
def list_img(request):
    image_name = predict.objects.get(name='4')
    # print(image_name)
    # for image in image_name:
    # print(image.img_input)
    image_item = image_name
    variables ={
    'carx':image_item
    }
    return render(request,'myapp/list_img.html',variables)
def haar_cascade_extact(img):
    img_origin = img
    img_cards =[]
    boxes =[]
    ID_cascade = cv2.CascadeClassifier("F:/project_2/opencv-haar-classifier-training/classifier1/cascade_.xml")
    for i in range(4):
        if i <3 and i >0:
            img = cv2.rotate(img, i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        IDcard = ID_cascade.detectMultiScale(gray, 1.1, 2)
        for (x,y,w,h) in IDcard:
            # cv2.rectangle(img,(x,y),(int(x+w*1.1),y+h),(255,0,0),2)
            boxes.append([x,y,w,h])
            roi_color = img[y:y+h, x:x+w]
            img_cards.append(roi_color)
    return img_origin, boxes, img_cards