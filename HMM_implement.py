# HMM for python layer
import numpy as np
import json
import torch
import string
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, tree,blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank
        self.tree = tree

    def forward(self, emission: torch.Tensor,to_string=True):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        oute =''
        index = torch.argmax(emission, dim=-1)
        for inde in emission:
            list_num = list_cha(oute,self.tree,self.labels)
            inde = inde*0.6+ list_num*0.4
            inde = torch.argmax(inde, dim=-1)  # [num_seq,]
            # inde = torch.unique_consecutive(inde, dim=-1)
            # predict(oute,'',self.tree)
            if inde != self.blank:
                oute += self.labels[inde[0]]
        if to_string:
            return oute
        else:
            return index
def printtree(Node,lever):
    # print(Node.data)
    f.write(str(Node.data)+'\n')
    if(Node.child!=[]):
        for i in Node.child:
            tab=''
            for num_tab in range(lever):
                tab=tab+'   '
            f.write(tab+'|'+'\n')
            f.write(tab+'---')
            printtree(i,lever+1)
def precent(root,lever):
    if root.child!=[]:
        for i in root.child:
            precent(i,lever+1)
    if lever>=1:
            key=str(root.parent.data.keys())[12:-3]
            num=root.parent.data[key]
            key1=str(root.data.keys())[12:-3]
            root.data[key1]/=num    
class Node:

    def __init__(self, data,child=None):
        # left child
        self.parent = None
        # right child
        self.child = []
        # node's value
        
        if child!=None:
            self.child.append(child)
        if data!=None:
            self.data = data
    # print function
    def PrintTree(self):
        print(self.data)
    def find(self,char):
        done=0
        stt =-1
        for j,i in enumerate(self.child):
            key=str(i.data.keys())[12:-3]
            if key==char:
                done=1
                stt=j
        if done==1:
            return True,stt
        else:
            return False,stt       
    def list_child(self, string):
        out =np.zeros((1,len(string)))
        for j,i in enumerate(self.child):
            key=str(i.data.keys())[12:-3]
            stt = np.where(string==key)
            out[stt]= int(list(i.data.values())[0])
        return out
contents=[]
def add(num,char,word,Node1,num_inc):
    if num>0:
        name=word[num_inc]
        num_inc+=1
        
        vitri1=Node1.find(name)[1]
        add(num-1,char,word,Node1.child[vitri1],num_inc)
       
    else:
        if Node1.find(char)[0]:
            vitri=Node1.find(char)[1]
            Node1.child[vitri].data[char]+=1
            
        else:
            new_node=Node({char:1})
            new_node.parent=Node1
            Node1.child.append(new_node)
            
with open("F:/project_2/vietnamese-namedb/girl_one_word.txt",'r',encoding='utf-8') as f:
    for line in f:
        # spli=line.split('*')
        content=line.strip('\n')
        # bytes(s,'utf-8').decode()
        # try:
        # content=content.split(' ')
        contents.append([i for i in content])
root=Node({'begin':len(contents)})
for i in contents:
    for num,char in enumerate(i) :
        add(num,char,i,root,0)
precent(root,0)
with open('log.txt','w+',encoding='utf-8') as f:
    printtree(root,0)
def json_save(root,lever,json):
    json_child ={}
    if(root.child!=[]):
        for i in root.child:
            json_child = json_save(i,lever+1,json_child)
            json.update({list(root.data.keys())[0]:[list(root.data.values())[0],json_child]})
    else:
        json.update(root.data)
    return json

def predict(stri,char,root):
    sum=[]
    for i in stri+char:
        try:
            vitri=root.find(i)
            more=root.child[vitri[1]].data[i]
            sum.append(more)
            root=root.child[vitri[1]]
        except:
            sum.append(1e-4)
    # print(sum)
def list_cha(stri,root,string):
    try:
        for i in stri[:-1]:
            vitri=root.find(i)
            root=root.child[vitri[1]]
        num = root.list_child(string)
    except:
        # print(string, i)
        num = np.zeros((1,len(string)))
    return num
# if __name__ == "__main__":
predict('Ý','',root)
json_val = {}
json_val = json_save(root,0, json_val)
# out_file = open("myfile.json", "w", encoding ='utf-8')
# json.dump(json_val, out_file, indent = 3)  
# out_file.close()
# print(json.dump(json_val))
f = open('myfile.json')
data =json.load(f)
root = Node({'begin':len(contents)})
# print(data)
def add_node(root,data,i):
    if i == "begin":
        root = Node({i:data[i][0]})
    if str(type(list(data[i][1].keys())))[8:-2]=='list':
        for j in list(data[i][1].keys()):
            if str(type(data[i][1][j]))[8:-2]=='list':
                root.child.append(Node({j:data[i][1][j][0]}))
                # print(0,{j:data[i][1][j][0]})
                add_node(root.child[-1],data[i][1],j)
            else:
                root.child.append(Node({j:data[i][1][j]}))
                # print(1,{j:data[i][1][j]})
    return root
root = add_node(root,data,'begin')
predict('huy','r',root)
decoder_ctc =  GreedyCTCDecoder(string.printable+'  ', root)
ran= np.random.randn(8,102)
output = decoder_ctc.forward(torch.tensor(ran),to_string=True)
# print('đầu ra là:',output)
