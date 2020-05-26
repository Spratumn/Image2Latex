#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_loader import *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64,(3,3),(1,1),(1,1))
        self.pool1 = nn.MaxPool2d((2,2), (2,2),(0,0))
        self.conv2 = nn.Conv2d(64, 128,(3,3),(1,1),(1,1))
        self.pool2 = nn.MaxPool2d((2,2),(2,2),(0,0))
        self.conv3 = nn.Conv2d(128, 256,(3,3),(1,1),(1,1))
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,256,(3,3),(1,1),(1,1))
        self.pool3 = nn.MaxPool2d((2,1),(2,1),(0,0))
        self.conv5 = nn.Conv2d(256,512,(3,3),(1,1),(1,1))
        self.conv5_bn = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d((1,2),(1,2),(0,0))
        self.conv6 = nn.Conv2d(512,512,(3,3),(1,1),(1,1))
        self.conv6_bn = nn.BatchNorm2d(512)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        #x = self.pool1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        #x = self.pool2(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.pool3(F.relu(self.conv4(x)))
        #x = self.pool3(x)
        x =self.pool4(F.relu(self.conv5_bn(self.conv5(x))))
        #x = self.pool4(x)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size,device):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size,batch_first=True)

    def forward(self, x):
        #请再这里实现Encoder的计算部分

        return outputs, (hn,cn)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x, hp, cp):
        # 在这里完成decode的中的计算	
        # P: log softmax of symbol scores -> (batch, output_size)
     
        return P, (hn, cn)


if __name__=="__main__":
    vocab = Vocab()
    data  = data_loader(vocab)
    image_train,formula_train,formulas,epoch_ended = data_loader.get_next_batch(data)
    B=2
    rand_image = torch.randn(B, 1, 256, 256)
    
    image_train = torch.from_numpy(image_train)
    input_image = torch.tensor(image_train, dtype=torch.float32)
    #input_image = rand_image
    #import pdb
    #pdb.set_trace()



    print("cnn")
    cnn=CNN()
    feature=cnn(input_image)
    # 2,512,32,32
    print(feature.shape)
    print("-"*20)
    print("mat to vector")
    # 在这里完成从feature map 到序列化向量的过程
    # feature = map2vector(feature)

    print(feature.shape)
    print("-"*20)
    print("encoder:")
    encode = Encoder(512,512,"cpu")
    encoder_outputs,(ehn,ecn) = encode(feature)  
    print("encoder hn:",ehn.shape)
    print("-"*20)

    print("decoder:")
    decode= Decoder(80,512,80,"cpu")

    token_start =torch.tensor([0,0])
    token_vector=torch.nn.functional.one_hot(token_start, 80)
    logits=[]
    preds=[]
    formulas=["",""]
    #pdb.set_trace()

    for i in range(0,B):
         token_numpy =token_start.cpu().numpy()
         formulas[i]+=vocab.idx2token[token_numpy[i]]

    for t in range(1,20):
        P,(dhn,dcn)=decode(token_vector.view(B,1,80).float(),ehn,ecn)
	logits.append(P)
	next_token = P.argmax(dim=1)
	for i in range(0,B):
             token_numpy =next_token.cpu().numpy()
             formulas[i]+=vocab.idx2token[token_numpy[i]]
        token_vector=torch.nn.functional.one_hot(next_token, 80)
	preds.append(next_token)
    #增加start_token的概率值 
    logits = [torch.zeros(logits[1].shape[0], logits[1].shape[1])] + logits
    # 讲start_token对应的位置设置为1
    logits[0][:, token_start] = 1
    # 列表变tensor
    logits = torch.stack(logits, dim=1)
    # 列表变tensor
    preds = torch.stack(preds, dim=1)
    #pdb.set_trace()
    print("-"*20)
    print(formulas)
