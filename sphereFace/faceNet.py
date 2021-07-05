import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import math

def myphi(x,m):
    x = x * m # m theta
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ] # taylor approx of cos(m*theta)

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B norm of x
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum), x @ ww (weight)
        cos_theta = cos_theta / torch.clamp(xlen.view(-1,1) * wlen.view(1,-1), min=1e-8)
        cos_theta = cos_theta.clamp(-1,1)

        # IMPLEMENT phi_theta, very slow
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = Variable(cos_theta.data.acos())
        k = (self.m*theta/3.14159265).floor()
        n_one = k*0.0 - 1
        phi_theta = (n_one**k) * cos_m_theta - 2*k

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)

        output = (cos_theta,phi_theta)
        return output


class CustomLoss(nn.Module):
    def __init__(self, gamma=0):
        super(CustomLoss, self).__init__()
        self.gamma = gamma
        # Parameters for computing loss function
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(Batch_size,1)

        # IMPLEMENT loss
        index = cos_theta.data * 0.0 #size=(Batch_size,Classnum)
        index.scatter_(1,target.data.view(-1,1),1) # copy 1 into index at the indices specified by target.data.view(-1,1)
        index = index.to(torch.bool) # convert to binary mask
        index = Variable(index) # one hot encoded class vector

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb) # torch.Size([128, 10574]), 128 is batch size
        # print(f"output.shape is : {output.shape}")

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()
        _, predictedLabel = torch.max(cos_theta.data, 1)
        predictedLabel = predictedLabel.view(-1, 1)
        accuracy = (predictedLabel.eq(target.data).cpu().sum().item() ) / float(target.size(0) )

        return loss, accuracy



class faceNet(nn.Module):
    def __init__(self,classnum=10574, feature=False, m = 1.35):
        super(faceNet, self).__init__()
        self.classnum = classnum
        self.feature = feature

        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.conv1_1bn = nn.BatchNorm2d(64)
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.conv1_2bn = nn.BatchNorm2d(64)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.conv1_3bn = nn.BatchNorm2d(64)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.conv2_1bn = nn.BatchNorm2d(128)
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.conv2_2bn = nn.BatchNorm2d(128)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.conv2_3bn = nn.BatchNorm2d(128)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.conv2_4bn = nn.BatchNorm2d(128)
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.conv2_5bn = nn.BatchNorm2d(128)
        self.relu2_5 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.conv3_1bn = nn.BatchNorm2d(256)
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.conv3_2bn = nn.BatchNorm2d(256)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.conv3_3bn = nn.BatchNorm2d(256)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.conv3_4bn = nn.BatchNorm2d(256)
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.conv3_5bn = nn.BatchNorm2d(256)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.conv3_6bn = nn.BatchNorm2d(256)
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.conv3_7bn = nn.BatchNorm2d(256)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.conv3_8bn = nn.BatchNorm2d(256)
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.conv3_9bn = nn.BatchNorm2d(256)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.conv4_1bn = nn.BatchNorm2d(512)
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.conv4_2bn = nn.BatchNorm2d(512)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.conv4_3bn = nn.BatchNorm2d(512)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.fc5_bn = nn.BatchNorm1d(512)
        self.fc6 = CustomLinear(in_features = 512,
                out_features = self.classnum, m=m)


    def forward(self, x):
        x = self.relu1_1(self.conv1_1bn(self.conv1_1(x)))
        x = x + self.relu1_3(self.conv1_3bn(self.conv1_3(self.relu1_2(self.conv1_2bn(self.conv1_2(x))))))

        x = self.relu2_1(self.conv2_1bn(self.conv2_1(x)))
        x = x + self.relu2_3(self.conv2_3bn(self.conv2_3(self.relu2_2(self.conv2_2bn(self.conv2_2(x))))))
        x = x + self.relu2_5(self.conv2_5bn(self.conv2_5(self.relu2_4(self.conv2_4bn(self.conv2_4(x))))))

        x = self.relu3_1(self.conv3_1bn(self.conv3_1(x)))
        x = x + self.relu3_3(self.conv3_3bn(self.conv3_3(self.relu3_2(self.conv3_2bn(self.conv3_2(x))))))
        x = x + self.relu3_5(self.conv3_5bn(self.conv3_5(self.relu3_4(self.conv3_4bn(self.conv3_4(x))))))
        x = x + self.relu3_7(self.conv3_7bn(self.conv3_7(self.relu3_6(self.conv3_6bn(self.conv3_6(x))))))
        x = x + self.relu3_9(self.conv3_9bn(self.conv3_9(self.relu3_8(self.conv3_8bn(self.conv3_8(x))))))

        x = self.relu4_1(self.conv4_1bn(self.conv4_1(x)))
        x = x + self.relu4_3(self.conv4_3bn(self.conv4_3(self.relu4_2(self.conv4_2bn(self.conv4_2(x))))))

        x = x.view(x.size(0),-1)
        x = self.fc5_bn(self.fc5(x))

        if self.feature:
            return x

        x = self.fc6(x)
        return x
