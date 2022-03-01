import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import OrderedDict
import cv2
import torchvision
import sys
sys.path.append("..")
from bcnn import BCNN
import pdb




device = 'cpu'
cot2 = BCNN(n_classes=200, pretrained=False)
state_dict = torch.load("/home/zhuyuan/zy_all/weblyFG-dataset-main/model/net2_step1_vgg16_best_epoch.pth")



from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
    new_state_dict[name] = v
cot2.load_state_dict(new_state_dict)
cot2.to(device)
# print(cot2)
pdb.set_trace()

for i, (name, module) in enumerate(cot2.features._modules.items()):
        print(i)
        print(name)
        print(module)


# for module in cot2.named_modules():
# 	print(module)


'''
model = BCNN(n_classes=200, pretrained=False)
params = model.state_dict()
for k,v in params.items():
    print(k)

print("***********************************")

cot2 = BCNN(n_classes=200, pretrained=False)
state_dict = torch.load("/home/zhuyuan/zy_all/weblyFG-dataset-main/model/net2_step1_vgg16_best_epoch.pth")
for k,v in state_dict.items():
    print(k)
'''


test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=448),
        torchvision.transforms.CenterCrop(size=448),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

# 
img_path = './Parakeet_Auklet_0026_795962.jpg'
img = Image.open(img_path)
# imgarray = np.array(img) / 255.0
imgarray = np.array(img)


# plt.figure(figsize=(8,8))
# plt.imshow(imgarray)
# plt.axis('off')
# plt.show()

# 
# pdb.set_trace()
img = test_transform(img).unsqueeze(0)
# img = test_transform(img).view(-1,3,448,448)
print(img.shape)

# 定义钩子函数，获取指定层名称的特征
activation = {} # 保存获取的输出
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

cot2.eval()
# 获取浅层特征
cot2.features[0].register_forward_hook(get_activation('28')) # 为28注册钩子
# cot2.features[0]:Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
_ = cot2(img)

bn3 = activation['28'] # 结果将保存在activation字典中
print(bn3.shape)

# plt.figure(figsize=(12,12))
# for i in range(64):
#     plt.subplot(8,8,i+1)
#     plt.imshow(bn3[0,i,:,:], cmap='gray')
#     plt.axis('off')
# plt.show()


#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
def draw_CAM(model, img, save_path, transform=None, visual_heatmap=True):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    
    # 获取模型输出的feature/score
    model.eval()
    
    features = model.features(img)
    bp_output = model.bilinear_pool(features)
    output = model.fc(bp_output)
    
 
    # 为了能读取到中间梯度定义的辅助函数
    #可以通过使用全局dict()或者全局list保存全部的，这里只保存一层的
    def extract(g):
        global features_grad
        features_grad = g
 
    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
    # pdb.set_trace()
 
    features.register_hook(extract)
    pred_class.backward() # 计算梯度
    # pdb.set_trace()
 
    grads = features_grad   # 获取梯度
 
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
 
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 68是最后一层feature的通道数
    for i in range(68):
        features[i, ...] *= pooled_grads[i, ...]
 
    # 以下部分
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 
    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
 
    img = cv2.imread('./Parakeet_Auklet_0026_795962.jpg')  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘

save_path = "./result2.jpg"
draw_CAM(cot2, img, save_path)
# https://blog.csdn.net/sinat_37532065/article/details/103362517



# 下面网址：不使用钩子，在模型定义中可视化特征图
# https://blog.csdn.net/weixin_40500230/article/details/93845890