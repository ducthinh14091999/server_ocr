import os
import string
device = "cpu"

img_dir = "./myapp/kie/images"
result_img_dir = "./myapp/kie/results/model"
raw_img_dir = "./myapp/kie/results/raw"
cropped_img_dir = "./myapp/kie/results/crop"

alphabet = string.printable+'ĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐÐÊÉẾÈỀẺỂẼỄẸỆÍÌỈĨỊÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢƯÚỨÙỪỦỬŨỮỤỰÝỲỶỸỴăâáắấàằầảẳẩãẵẫạặậđêéếèềẻểẽễẹệíìỉĩịôơóốớòồờỏổởõỗỡọộợưúứùừủửũữụựýỳỷỹỵ'
node_labels = ['khac','other','ID','name','day of birth','male/female','nationality','native place','place living']


text_detection_weights_path = "./myapp/kie/weights/text_detect/craft_mlt_25k_1.pth"
saliency_weight_path = "./myapp/kie/weights/saliency/u2netp.pth"
kie_weight_path = "./myapp/kie/weights/kie/GCN_model.pkl"

saliency_ths = 0.5
score_ths = 0.82
get_max = True  # get max score / filter predicted categories by score threshold
merge_text = True  # if True, concatenate text contents from left to right
infer_batch_vietocr = True  # inference with batch
visualize = False
