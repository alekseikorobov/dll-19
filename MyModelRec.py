import os,errno
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class CRNN(nn.Module):

    def __init__(self,cnn_output_height, gru_hidden_size, gru_num_layers, num_classes):
        super(CRNN, self).__init__()
        self.num_classes = num_classes
        ### сверточная часть
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.norm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2)
        self.norm2 = nn.InstanceNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.norm3 = nn.InstanceNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2)
        self.norm4 = nn.InstanceNorm2d(64)
        self.gru_input_size = cnn_output_height * 64
        ### рекурентная часть
        self.gru_name = 'LSTM'
        #self.gru = nn.GRU(self.gru_input_size, gru_hidden_size, gru_num_layers, batch_first=True, bidirectional=True)
        self.gru = nn.LSTM(self.gru_input_size, gru_hidden_size, gru_num_layers, batch_first=True, bidirectional=True)
        ### классифицирующая часть
        self.fc = nn.Linear(gru_hidden_size * 2, num_classes)
        self.version='v0'

    def forward(self, x):
        batch_size = x.shape[0]
        #print(f'{x.shape=}') # torch.Size([3, 3, 18, 50])
        ### сверточная часть
        out = self.conv1(x)
        #print(f'conv1 {out.shape=}') #out.shape=torch.Size([3, 32, 16, 48])
        out = self.norm1(out)
        #print(f'norm1 {out.shape=}') #out.shape=torch.Size([3, 32, 16, 48])
        out = F.leaky_relu(out)
        out = self.conv2(out)
        #print(f'conv2 {out.shape=}') #out.shape=torch.Size([3, 32, 7, 23])
        out = self.norm2(out)
        #print(f'norm2 {out.shape=}') #out.shape=torch.Size([3, 32, 7, 23])
        out = F.leaky_relu(out)
        out = self.conv3(out)
        #print(f'conv3 {out.shape=}') #out.shape=torch.Size([3, 64, 5, 21])
        out = self.norm3(out)
        #print(f'norm3 {out.shape=}') #out.shape=torch.Size([3, 64, 5, 21])
        out = F.leaky_relu(out)
        out = self.conv4(out)
        #print(f'conv4 {out.shape=}') #out.shape=torch.Size([3, 64, 2, 10])
        out = self.norm4(out)
        #print(f'{out.shape=}') #out.shape=torch.Size([3, 64, 2, 10])
        out = F.leaky_relu(out)
        #print(f'{out.shape=}') #out.shape=torch.Size([3, 64, 2, 10])
        ### нарезаем картинку на части
        out = out.permute(0, 3, 2, 1)
        #print(f'{out.shape=}') #out.shape=torch.Size([3, 10, 2, 64])
        out = out.reshape(batch_size, -1, self.gru_input_size)
        #print(f'{out.shape=}') #out.shape=torch.Size([3, 5, 256])
        ### прогоняемполучившуюся последовательность через рекрентную часть
        out, _ = self.gru(out)
        #print(f'{out.shape=}') #out.shape=torch.Size([3, 5, 256])
        ### переводим в вероятности выход из рекурентной части
        out = torch.stack([F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])])
        return out


class BidirectionalLSTM(nn.Module):

    def __init__(self, gru_input_size, gru_hidden_size, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(gru_input_size, gru_hidden_size, bidirectional=True)
        self.embedding = nn.Linear(gru_hidden_size * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class CRNN_v1(nn.Module):

    #def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
    def __init__(self, imgH, in_channels, nclass, gru_size, leakyRelu=False):
        super(CRNN_v1, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.num_classes = nclass
        # 1x32x128
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 64x16x64
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 128x8x32
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        # 256x4x16
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        # 512x2x16
        self.conv5 = nn.Conv2d(512, 512, 2, 1, 0)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(True)

        # 512x1x16
        self.gru_name = 'LSTM'
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, gru_size, gru_size),
            BidirectionalLSTM(gru_size, gru_size, nclass))
        
        self.version='v1'


    def forward(self, input):
        # conv features
        x = self.pool1(self.relu1(self.conv1(input)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3_2(self.conv3_2(self.relu3_1(self.bn3(self.conv3_1(x))))))
        x = self.pool4(self.relu4_2(self.conv4_2(self.relu4_1(self.bn4(self.conv4_1(x))))))
        conv = self.relu5(self.bn5(self.conv5(x)))
        # print(conv.size())

        b, c, h, w = conv.size()
        #print(f'{b, c, h, w=}')
        assert h == 1, f"the height of conv must be 1, now {h=}"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


class CRNN_v2(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN_v2, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        # 1x32x128
        self.conv1_1 = nn.Conv2d(nc, 32, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.ReLU(True)

        self.conv1_2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 64x16x64
        self.conv2_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU(True)

        self.conv2_2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 128x8x32
        self.conv3_1 = nn.Conv2d(128, 96, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(96)
        self.relu3_1 = nn.ReLU(True)

        self.conv3_2 = nn.Conv2d(96, 192, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(192)
        self.relu3_2 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        # 192x4x32
        self.conv4_1 = nn.Conv2d(192, 128, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.relu4_1 = nn.ReLU(True)
        self.conv4_2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.relu4_2 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        # 256x2x32
        self.bn5 = nn.BatchNorm2d(256)


        # 256x2x32
        self.gru_name = 'LSTM'
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        
        self.version='v2'


    def forward(self, input):
        # conv features
        x = self.pool1(self.relu1_2(self.bn1_2(self.conv1_2(self.relu1_1(self.bn1_1(self.conv1_1(input)))))))
        x = self.pool2(self.relu2_2(self.bn2_2(self.conv2_2(self.relu2_1(self.bn2_1(self.conv2_1(x)))))))
        x = self.pool3(self.relu3_2(self.bn3_2(self.conv3_2(self.relu3_1(self.bn3_1(self.conv3_1(x)))))))
        x = self.pool4(self.relu4_2(self.bn4_2(self.conv4_2(self.relu4_1(self.bn4_1(self.conv4_1(x)))))))
        conv = self.bn5(x)
        # print(conv.size())

        b, c, h, w = conv.size()
        assert h == 2, "the height of conv must be 2"
        conv = conv.reshape([b,c*h,w])
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


def conv3x3(nIn, nOut, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d( nIn, nOut, kernel_size=3, stride=stride, padding=1, bias=False )


class basic_res_block(nn.Module):

    def __init__(self, nIn, nOut, stride=1, downsample=None):
        super( basic_res_block, self ).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3( nIn, nOut, stride )
        m['bn1'] = nn.BatchNorm2d( nOut )
        m['relu1'] = nn.ReLU( inplace=True )
        m['conv2'] = conv3x3( nOut, nOut )
        m['bn2'] = nn.BatchNorm2d( nOut )
        self.group1 = nn.Sequential( m )

        self.relu = nn.Sequential( nn.ReLU( inplace=True ) )
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample( x )
        else:
            residual = x
        out = self.group1( x ) + residual
        out = self.relu( out )
        return out


class CRNN_res(nn.Module):

    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN_res, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.conv1 = nn.Conv2d(nc, 64, 3, 1, 1)
        self.relu1 = nn.ReLU(True)
        self.res1 = basic_res_block(64, 64)
        # 1x32x128

        down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(128))
        self.res2_1 = basic_res_block( 64, 128, 2, down1 )
        self.res2_2 = basic_res_block(128,128)
        # 64x16x64

        down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(256))
        self.res3_1 = basic_res_block(128, 256, 2, down2)
        self.res3_2 = basic_res_block(256, 256)
        self.res3_3 = basic_res_block(256, 256)
        # 128x8x32

        down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=(2, 1), bias=False),nn.BatchNorm2d(512))
        self.res4_1 = basic_res_block(256, 512, (2, 1), down3)
        self.res4_2 = basic_res_block(512, 512)
        self.res4_3 = basic_res_block(512, 512)
        # 256x4x16

        self.pool = nn.AvgPool2d((2, 2), (2, 1), (0, 1))
        # 512x2x16

        self.conv5 = nn.Conv2d(512, 512, 2, 1, 0)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(True)
        # 512x1x16

        self.gru_name = 'LSTM'
        
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.version='v3'

    def forward(self, input):
        # conv features
        x = self.res1(self.relu1(self.conv1(input)))
        x = self.res2_2(self.res2_1(x))
        x = self.res3_3(self.res3_2(self.res3_1(x)))
        x = self.res4_3(self.res4_2(self.res4_1(x)))
        x = self.pool(x)
        conv = self.relu5(self.bn5(self.conv5(x)))
        # print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output



    

def load_model(model:CRNN, model_path, device)->CRNN:
    #model = CRNN(cnn_output_height, gru_hidden_size, gru_num_layers, num_classes)
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict['model'])
    return model

def mkdirs(newdir):
    """
    make directory with parent path
    :param newdir: target path
    """
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise
        
def save_model(model:CRNN, epoch, lr, optimzer, save_dir, name = None):
    
    if name is None:
        name = f'CRNN_{model.gru_name}_{model.version}_{epoch}.pth'
    save_path = os.path.join(save_dir, name)
    
    if not os.path.exists(save_dir):
        mkdirs(save_dir)
        
    print(f'Saving to {save_path}.')
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict()
        # 'optimizer': optimzer.state_dict()
    }
    torch.save(state_dict, save_path)
    return save_path

def remove_old_model(model, epoch, save_freq, save_dir):
    old_epoch = epoch - save_freq
    
    remove_path = os.path.join(save_dir, f'CRNN_{model.gru_name}_{model.version}_{old_epoch}.pth')
    
    if os.path.exists(remove_path):
        print(f'{remove_path=}.')
        os.remove(remove_path)