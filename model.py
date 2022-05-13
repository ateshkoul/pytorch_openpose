import torch
from collections import OrderedDict
import pdb
import torch
import torch.nn as nn
import numpy as np
# for face
import torch.nn.functional as F

# old implementation not optimised for jit
#def make_layers(block, no_relu_layers):
#    layers = []
#    for layer_name, v in block.items():
#        if 'pool' in layer_name:
#            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
#                                    padding=v[2])
#            layers.append((layer_name, layer))
#        else:
#            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
#                               kernel_size=v[2], stride=v[3],
#                               padding=v[4])
#            layers.append((layer_name, conv2d))
#            if layer_name not in no_relu_layers:
#                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))
#
#    return nn.Sequential(OrderedDict(layers))
	
# this implementation reduces the prediction time (sometimes) but increases the time 
# in making the model by like 8-10 sec instead of like 2 sec.
def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            #layer = torch.jit.trace(nn.MaxPool2d(kernel_size=v[0], stride=v[1],
            #                        padding=v[2]),torch.randn(1,v[0],28,28))
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            #pdb.set_trace()
            #conv2d = torch.jit.trace(nn.Conv2d(in_channels=v[0], out_channels=v[1],
            #       kernel_size=v[2], stride=v[3],
            #       padding=v[4]),torch.randn(1,v[0],28,28))
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))

def load_weights(weight_file):
    if weight_file == None:
        return
    #pdb.set_trace()
    try:
        #weights_dict = np.load(weight_file).item()
        weights_dict = torch.load(weight_file)
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

# a huge amount of time is devoted to making the layers itself in jit mode (approx 8 sec)
# however this is done only once at the begining of the experiment. 
class bodypose_model(nn.Module):
    def __init__(self):
        super(bodypose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict({'conv1_1': [3, 64, 3, 1, 1],
                  'conv1_2': [64, 64, 3, 1, 1],
                  'pool1_stage1': [2, 2, 0],
                  'conv2_1': [64, 128, 3, 1, 1],
                  'conv2_2': [128, 128, 3, 1, 1],
                  'pool2_stage1': [2, 2, 0],
                  'conv3_1': [128, 256, 3, 1, 1],
                  'conv3_2': [256, 256, 3, 1, 1],
                  'conv3_3': [256, 256, 3, 1, 1],
                  'conv3_4': [256, 256, 3, 1, 1],
                  'pool3_stage1': [2, 2, 0],
                  'conv4_1': [256, 512, 3, 1, 1],
                  'conv4_2': [512, 512, 3, 1, 1],
                  'conv4_3_CPM': [512, 256, 3, 1, 1],
                  'conv4_4_CPM': [256, 128, 3, 1, 1]})


        # Stage 1
        block1_1 = OrderedDict({'conv5_1_CPM_L1': [128, 128, 3, 1, 1],
                    'conv5_2_CPM_L1': [128, 128, 3, 1, 1],
                    'conv5_3_CPM_L1': [128, 128, 3, 1, 1],
                    'conv5_4_CPM_L1': [128, 512, 1, 1, 0],
                    'conv5_5_CPM_L1': [512, 38, 1, 1, 0]})

        block1_2 = OrderedDict({'conv5_1_CPM_L2': [128, 128, 3, 1, 1],
                    'conv5_2_CPM_L2': [128, 128, 3, 1, 1],
                    'conv5_3_CPM_L2': [128, 128, 3, 1, 1],
                    'conv5_4_CPM_L2': [128, 512, 1, 1, 0],
                    'conv5_5_CPM_L2': [512, 19, 1, 1, 0]})
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict({
                'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3],
                'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]})

            blocks['block%d_2' % i] = OrderedDict({
                'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3],
                'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]})

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']


    def forward(self, x):

        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2

class handpose_model(nn.Module):
    def __init__(self):
        super(handpose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',\
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        # stage 1
        block1_0 = OrderedDict({
            'conv1_1': [3, 64, 3, 1, 1],
            'conv1_2': [64, 64, 3, 1, 1],
            'pool1_stage1': [2, 2, 0],
            'conv2_1': [64, 128, 3, 1, 1],
            'conv2_2': [128, 128, 3, 1, 1],
            'pool2_stage1': [2, 2, 0],
            'conv3_1': [128, 256, 3, 1, 1],
            'conv3_2': [256, 256, 3, 1, 1],
            'conv3_3': [256, 256, 3, 1, 1],
            'conv3_4': [256, 256, 3, 1, 1],
            'pool3_stage1': [2, 2, 0],
            'conv4_1': [256, 512, 3, 1, 1],
            'conv4_2': [512, 512, 3, 1, 1],
            'conv4_3': [512, 512, 3, 1, 1],
            'conv4_4': [512, 512, 3, 1, 1],
            'conv5_1': [512, 512, 3, 1, 1],
            'conv5_2': [512, 512, 3, 1, 1],
            'conv5_3_CPM': [512, 128, 3, 1, 1]})

        block1_1 = OrderedDict({
            'conv6_1_CPM': [128, 512, 1, 1, 0],
            'conv6_2_CPM': [512, 22, 1, 1, 0]
        })

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict({
                'Mconv1_stage%d' % i: [150, 128, 7, 1, 3],
                'Mconv2_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d' % i: [128, 22, 1, 1, 0]})

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6

class facepose_model(nn.Module):

    
    def __init__(self, weight_file):
        super(facepose_model, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)
		# was for jit
        #__weights_dict = load_weights(torch.nn.Parameter(weight_file))
        
        self.conv1_1 = self.__conv(2, name='conv1_1', weight_file=weight_file,in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        #pdb.set_trace()
        self.conv1_2 = self.__conv(2, name='conv1_2', weight_file=weight_file,in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2_1 = self.__conv(2, name='conv2_1', weight_file=weight_file,in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv2_2 = self.__conv(2, name='conv2_2', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_1 = self.__conv(2, name='conv3_1', weight_file=weight_file,in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_2 = self.__conv(2, name='conv3_2', weight_file=weight_file,in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_3 = self.__conv(2, name='conv3_3', weight_file=weight_file,in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv3_4 = self.__conv(2, name='conv3_4', weight_file=weight_file,in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_1 = self.__conv(2, name='conv4_1', weight_file=weight_file,in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_2 = self.__conv(2, name='conv4_2', weight_file=weight_file,in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_3 = self.__conv(2, name='conv4_3', weight_file=weight_file,in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv4_4 = self.__conv(2, name='conv4_4', weight_file=weight_file,in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_1 = self.__conv(2, name='conv5_1', weight_file=weight_file,in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_2 = self.__conv(2, name='conv5_2', weight_file=weight_file,in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv5_3_CPM = self.__conv(2, name='conv5_3_CPM', weight_file=weight_file,in_channels=512, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.conv6_1_CPM = self.__conv(2, name='conv6_1_CPM', weight_file=weight_file,in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.conv6_2_CPM = self.__conv(2, name='conv6_2_CPM', weight_file=weight_file,in_channels=512, out_channels=71, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.Mconv1_stage2 = self.__conv(2, name='Mconv1_stage2', weight_file=weight_file,in_channels=199, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv2_stage2 = self.__conv(2, name='Mconv2_stage2', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv3_stage2 = self.__conv(2, name='Mconv3_stage2', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv4_stage2 = self.__conv(2, name='Mconv4_stage2', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv5_stage2 = self.__conv(2, name='Mconv5_stage2', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv6_stage2 = self.__conv(2, name='Mconv6_stage2', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.Mconv7_stage2 = self.__conv(2, name='Mconv7_stage2', weight_file=weight_file,in_channels=128, out_channels=71, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.Mconv1_stage3 = self.__conv(2, name='Mconv1_stage3', weight_file=weight_file,in_channels=199, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv2_stage3 = self.__conv(2, name='Mconv2_stage3', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv3_stage3 = self.__conv(2, name='Mconv3_stage3', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv4_stage3 = self.__conv(2, name='Mconv4_stage3', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv5_stage3 = self.__conv(2, name='Mconv5_stage3', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv6_stage3 = self.__conv(2, name='Mconv6_stage3', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.Mconv7_stage3 = self.__conv(2, name='Mconv7_stage3', weight_file=weight_file,in_channels=128, out_channels=71, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.Mconv1_stage4 = self.__conv(2, name='Mconv1_stage4', weight_file=weight_file,in_channels=199, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv2_stage4 = self.__conv(2, name='Mconv2_stage4', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv3_stage4 = self.__conv(2, name='Mconv3_stage4', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv4_stage4 = self.__conv(2, name='Mconv4_stage4', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv5_stage4 = self.__conv(2, name='Mconv5_stage4', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv6_stage4 = self.__conv(2, name='Mconv6_stage4', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.Mconv7_stage4 = self.__conv(2, name='Mconv7_stage4', weight_file=weight_file,in_channels=128, out_channels=71, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.Mconv1_stage5 = self.__conv(2, name='Mconv1_stage5', weight_file=weight_file,in_channels=199, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv2_stage5 = self.__conv(2, name='Mconv2_stage5', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv3_stage5 = self.__conv(2, name='Mconv3_stage5', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv4_stage5 = self.__conv(2, name='Mconv4_stage5', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv5_stage5 = self.__conv(2, name='Mconv5_stage5', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv6_stage5 = self.__conv(2, name='Mconv6_stage5', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.Mconv7_stage5 = self.__conv(2, name='Mconv7_stage5', weight_file=weight_file,in_channels=128, out_channels=71, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.Mconv1_stage6 = self.__conv(2, name='Mconv1_stage6', weight_file=weight_file,in_channels=199, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv2_stage6 = self.__conv(2, name='Mconv2_stage6', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv3_stage6 = self.__conv(2, name='Mconv3_stage6', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv4_stage6 = self.__conv(2, name='Mconv4_stage6', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv5_stage6 = self.__conv(2, name='Mconv5_stage6', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(7, 7), stride=(1, 1), groups=1, bias=True)
        self.Mconv6_stage6 = self.__conv(2, name='Mconv6_stage6', weight_file=weight_file,in_channels=128, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.Mconv7_stage6 = self.__conv(2, name='Mconv7_stage6', weight_file=weight_file,in_channels=128, out_channels=71, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)

    def forward(self, x):
        conv1_1_pad     = F.pad(x, (1, 1, 1, 1))
        conv1_1         = self.conv1_1(conv1_1_pad)
        conv1_1_re      = F.relu(conv1_1)
        conv1_2_pad     = F.pad(conv1_1_re, (1, 1, 1, 1))
        conv1_2         = self.conv1_2(conv1_2_pad)
        conv1_2_re      = F.relu(conv1_2)
        pool1_pad       = F.pad(conv1_2_re, (0, 1, 0, 1), value=float('-inf'))
        pool1           = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2_1_pad     = F.pad(pool1, (1, 1, 1, 1))
        conv2_1         = self.conv2_1(conv2_1_pad)
        conv2_1_re      = F.relu(conv2_1)
        conv2_2_pad     = F.pad(conv2_1_re, (1, 1, 1, 1))
        conv2_2         = self.conv2_2(conv2_2_pad)
        conv2_2_re      = F.relu(conv2_2)
        pool2_pad       = F.pad(conv2_2_re, (0, 1, 0, 1), value=float('-inf'))
        pool2           = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv3_1_pad     = F.pad(pool2, (1, 1, 1, 1))
        conv3_1         = self.conv3_1(conv3_1_pad)
        conv3_1_re      = F.relu(conv3_1)
        conv3_2_pad     = F.pad(conv3_1_re, (1, 1, 1, 1))
        conv3_2         = self.conv3_2(conv3_2_pad)
        conv3_2_re      = F.relu(conv3_2)
        conv3_3_pad     = F.pad(conv3_2_re, (1, 1, 1, 1))
        conv3_3         = self.conv3_3(conv3_3_pad)
        conv3_3_re      = F.relu(conv3_3)
        conv3_4_pad     = F.pad(conv3_3_re, (1, 1, 1, 1))
        conv3_4         = self.conv3_4(conv3_4_pad)
        conv3_4_re      = F.relu(conv3_4)
        pool3_pad       = F.pad(conv3_4_re, (0, 1, 0, 1), value=float('-inf'))
        pool3           = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv4_1_pad     = F.pad(pool3, (1, 1, 1, 1))
        conv4_1         = self.conv4_1(conv4_1_pad)
        conv4_1_re      = F.relu(conv4_1)
        conv4_2_pad     = F.pad(conv4_1_re, (1, 1, 1, 1))
        conv4_2         = self.conv4_2(conv4_2_pad)
        conv4_2_re      = F.relu(conv4_2)
        conv4_3_pad     = F.pad(conv4_2_re, (1, 1, 1, 1))
        conv4_3         = self.conv4_3(conv4_3_pad)
        conv4_3_re      = F.relu(conv4_3)
        conv4_4_pad     = F.pad(conv4_3_re, (1, 1, 1, 1))
        conv4_4         = self.conv4_4(conv4_4_pad)
        conv4_4_re      = F.relu(conv4_4)
        conv5_1_pad     = F.pad(conv4_4_re, (1, 1, 1, 1))
        conv5_1         = self.conv5_1(conv5_1_pad)
        conv5_1_re      = F.relu(conv5_1)
        conv5_2_pad     = F.pad(conv5_1_re, (1, 1, 1, 1))
        conv5_2         = self.conv5_2(conv5_2_pad)
        conv5_2_re      = F.relu(conv5_2)
        conv5_3_CPM_pad = F.pad(conv5_2_re, (1, 1, 1, 1))
        conv5_3_CPM     = self.conv5_3_CPM(conv5_3_CPM_pad)
        conv5_3_CPM_re  = F.relu(conv5_3_CPM)
        conv6_1_CPM     = self.conv6_1_CPM(conv5_3_CPM_re)
        conv6_1_CPM_re  = F.relu(conv6_1_CPM)
        conv6_2_CPM     = self.conv6_2_CPM(conv6_1_CPM_re)
        features_in_stage_2 = torch.cat((conv6_2_CPM, conv5_3_CPM_re), 1)
        Mconv1_stage2_pad = F.pad(features_in_stage_2, (3, 3, 3, 3))
        Mconv1_stage2   = self.Mconv1_stage2(Mconv1_stage2_pad)
        Mconv1_stage2_re = F.relu(Mconv1_stage2)
        Mconv2_stage2_pad = F.pad(Mconv1_stage2_re, (3, 3, 3, 3))
        Mconv2_stage2   = self.Mconv2_stage2(Mconv2_stage2_pad)
        Mconv2_stage2_re = F.relu(Mconv2_stage2)
        Mconv3_stage2_pad = F.pad(Mconv2_stage2_re, (3, 3, 3, 3))
        Mconv3_stage2   = self.Mconv3_stage2(Mconv3_stage2_pad)
        Mconv3_stage2_re = F.relu(Mconv3_stage2)
        Mconv4_stage2_pad = F.pad(Mconv3_stage2_re, (3, 3, 3, 3))
        Mconv4_stage2   = self.Mconv4_stage2(Mconv4_stage2_pad)
        Mconv4_stage2_re = F.relu(Mconv4_stage2)
        Mconv5_stage2_pad = F.pad(Mconv4_stage2_re, (3, 3, 3, 3))
        Mconv5_stage2   = self.Mconv5_stage2(Mconv5_stage2_pad)
        Mconv5_stage2_re = F.relu(Mconv5_stage2)
        Mconv6_stage2   = self.Mconv6_stage2(Mconv5_stage2_re)
        Mconv6_stage2_re = F.relu(Mconv6_stage2)
        Mconv7_stage2   = self.Mconv7_stage2(Mconv6_stage2_re)
        features_in_stage_3 = torch.cat((Mconv7_stage2, conv5_3_CPM_re), 1)
        Mconv1_stage3_pad = F.pad(features_in_stage_3, (3, 3, 3, 3))
        Mconv1_stage3   = self.Mconv1_stage3(Mconv1_stage3_pad)
        Mconv1_stage3_re = F.relu(Mconv1_stage3)
        Mconv2_stage3_pad = F.pad(Mconv1_stage3_re, (3, 3, 3, 3))
        Mconv2_stage3   = self.Mconv2_stage3(Mconv2_stage3_pad)
        Mconv2_stage3_re = F.relu(Mconv2_stage3)
        Mconv3_stage3_pad = F.pad(Mconv2_stage3_re, (3, 3, 3, 3))
        Mconv3_stage3   = self.Mconv3_stage3(Mconv3_stage3_pad)
        Mconv3_stage3_re = F.relu(Mconv3_stage3)
        Mconv4_stage3_pad = F.pad(Mconv3_stage3_re, (3, 3, 3, 3))
        Mconv4_stage3   = self.Mconv4_stage3(Mconv4_stage3_pad)
        Mconv4_stage3_re = F.relu(Mconv4_stage3)
        Mconv5_stage3_pad = F.pad(Mconv4_stage3_re, (3, 3, 3, 3))
        Mconv5_stage3   = self.Mconv5_stage3(Mconv5_stage3_pad)
        Mconv5_stage3_re = F.relu(Mconv5_stage3)
        Mconv6_stage3   = self.Mconv6_stage3(Mconv5_stage3_re)
        Mconv6_stage3_re = F.relu(Mconv6_stage3)
        Mconv7_stage3   = self.Mconv7_stage3(Mconv6_stage3_re)
        features_in_stage_4 = torch.cat((Mconv7_stage3, conv5_3_CPM_re), 1)
        Mconv1_stage4_pad = F.pad(features_in_stage_4, (3, 3, 3, 3))
        Mconv1_stage4   = self.Mconv1_stage4(Mconv1_stage4_pad)
        Mconv1_stage4_re = F.relu(Mconv1_stage4)
        Mconv2_stage4_pad = F.pad(Mconv1_stage4_re, (3, 3, 3, 3))
        Mconv2_stage4   = self.Mconv2_stage4(Mconv2_stage4_pad)
        Mconv2_stage4_re = F.relu(Mconv2_stage4)
        Mconv3_stage4_pad = F.pad(Mconv2_stage4_re, (3, 3, 3, 3))
        Mconv3_stage4   = self.Mconv3_stage4(Mconv3_stage4_pad)
        Mconv3_stage4_re = F.relu(Mconv3_stage4)
        Mconv4_stage4_pad = F.pad(Mconv3_stage4_re, (3, 3, 3, 3))
        Mconv4_stage4   = self.Mconv4_stage4(Mconv4_stage4_pad)
        Mconv4_stage4_re = F.relu(Mconv4_stage4)
        Mconv5_stage4_pad = F.pad(Mconv4_stage4_re, (3, 3, 3, 3))
        Mconv5_stage4   = self.Mconv5_stage4(Mconv5_stage4_pad)
        Mconv5_stage4_re = F.relu(Mconv5_stage4)
        Mconv6_stage4   = self.Mconv6_stage4(Mconv5_stage4_re)
        Mconv6_stage4_re = F.relu(Mconv6_stage4)
        Mconv7_stage4   = self.Mconv7_stage4(Mconv6_stage4_re)
        features_in_stage_5 = torch.cat((Mconv7_stage4, conv5_3_CPM_re), 1)
        Mconv1_stage5_pad = F.pad(features_in_stage_5, (3, 3, 3, 3))
        Mconv1_stage5   = self.Mconv1_stage5(Mconv1_stage5_pad)
        Mconv1_stage5_re = F.relu(Mconv1_stage5)
        Mconv2_stage5_pad = F.pad(Mconv1_stage5_re, (3, 3, 3, 3))
        Mconv2_stage5   = self.Mconv2_stage5(Mconv2_stage5_pad)
        Mconv2_stage5_re = F.relu(Mconv2_stage5)
        Mconv3_stage5_pad = F.pad(Mconv2_stage5_re, (3, 3, 3, 3))
        Mconv3_stage5   = self.Mconv3_stage5(Mconv3_stage5_pad)
        Mconv3_stage5_re = F.relu(Mconv3_stage5)
        Mconv4_stage5_pad = F.pad(Mconv3_stage5_re, (3, 3, 3, 3))
        Mconv4_stage5   = self.Mconv4_stage5(Mconv4_stage5_pad)
        Mconv4_stage5_re = F.relu(Mconv4_stage5)
        Mconv5_stage5_pad = F.pad(Mconv4_stage5_re, (3, 3, 3, 3))
        Mconv5_stage5   = self.Mconv5_stage5(Mconv5_stage5_pad)
        Mconv5_stage5_re = F.relu(Mconv5_stage5)
        Mconv6_stage5   = self.Mconv6_stage5(Mconv5_stage5_re)
        Mconv6_stage5_re = F.relu(Mconv6_stage5)
        Mconv7_stage5   = self.Mconv7_stage5(Mconv6_stage5_re)
        features_in_stage_6 = torch.cat((Mconv7_stage5, conv5_3_CPM_re), 1)
        Mconv1_stage6_pad = F.pad(features_in_stage_6, (3, 3, 3, 3))
        Mconv1_stage6   = self.Mconv1_stage6(Mconv1_stage6_pad)
        Mconv1_stage6_re = F.relu(Mconv1_stage6)
        Mconv2_stage6_pad = F.pad(Mconv1_stage6_re, (3, 3, 3, 3))
        Mconv2_stage6   = self.Mconv2_stage6(Mconv2_stage6_pad)
        Mconv2_stage6_re = F.relu(Mconv2_stage6)
        Mconv3_stage6_pad = F.pad(Mconv2_stage6_re, (3, 3, 3, 3))
        Mconv3_stage6   = self.Mconv3_stage6(Mconv3_stage6_pad)
        Mconv3_stage6_re = F.relu(Mconv3_stage6)
        Mconv4_stage6_pad = F.pad(Mconv3_stage6_re, (3, 3, 3, 3))
        Mconv4_stage6   = self.Mconv4_stage6(Mconv4_stage6_pad)
        Mconv4_stage6_re = F.relu(Mconv4_stage6)
        Mconv5_stage6_pad = F.pad(Mconv4_stage6_re, (3, 3, 3, 3))
        Mconv5_stage6   = self.Mconv5_stage6(Mconv5_stage6_pad)
        Mconv5_stage6_re = F.relu(Mconv5_stage6)
        Mconv6_stage6   = self.Mconv6_stage6(Mconv5_stage6_re)
        Mconv6_stage6_re = F.relu(Mconv6_stage6)
        Mconv7_stage6   = self.Mconv7_stage6(Mconv6_stage6_re)
        return Mconv7_stage6


    @staticmethod
    def __conv(dim, name,weight_file,**kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()
        __weights_dict = load_weights(weight_file)
        #pdb.set_trace()
        #layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        layer.state_dict()['weight'].copy_(__weights_dict[name+'.weight'])
        if 'bias' in name:
            layer.state_dict()['bias'].copy_(__weights_dict[name+'.bias'])
        #if 'bias' in __weights_dict[name]:
        #    layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        #pdb.set_trace()
        return layer
#bodypose_model = torch.jit.script(bodypose_model())