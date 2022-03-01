# 使用的是grad-cam     cam仅适用于网络中包含全局平均池化的情况    https://www.zhihu.com/question/274926848
# coding: utf-8
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.autograd as autograd
import torchvision.transforms 
import sys
sys.path.append("..")
from bcnn import BCNN
import pdb

 
 
# 训练过的模型路径
# resume_path = r"D:\TJU\GBDB\set113\cross_validation\test1\epoch_0257_checkpoint.pth.tar"
# 输入图像路径
single_img_path = './Parakeet_Auklet_0026_795962.jpg'
# 绘制的热力图存储路径
save_path = "./result_f2.jpg"
 
# 网络层的层名列表, 需要根据实际使用网络进行修改
layers_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8','9', '10', '11', '12', '13', '14', '15', '16', '17','18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
# 指定层名
out_layer_name = "5"
 
features_grad = 0
 
 
# 为了读取模型中间参数变量的梯度而定义的辅助函数
def extract(g):
    global features_grad
    features_grad = g
 
 
def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=True, out_layer=None):
    """
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    """
    # 读取图像并预处理
    # global layer2
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img).cuda()
    img = img.unsqueeze(0)  # (1, 3, 448, 448)
    # pdb.set_trace()
 
    # model转为eval模式
    model.eval()
 
    # 获取模型层的字典
    layers_dict = {layers_names[i]: None for i in range(len(layers_names))}
    
    for i, (name, module) in enumerate(model.features._modules.items()):
        layers_dict[layers_names[i]] = module
 
    # 遍历模型的每一层, 获得指定层的输出特征图
    # features: 指定层输出的特征图, features_flatten: 为继续完成前端传播而设置的变量
    # pdb.set_trace()
    features = img
    start_flatten = False
    features_flatten = None
    for name, layer in layers_dict.items():
        if name != out_layer and start_flatten is False:    # 指定层之前
            features = layer(features)
        elif name == out_layer and start_flatten is False:  # 指定层
            features = layer(features)
            start_flatten = True
        else:   # 指定层之后
            if features_flatten is None:
                features_flatten = layer(features)
            else:
                features_flatten = layer(features_flatten)
 
    # pdb.set_trace()
    # features_flatten = torch.flatten(features_flatten, 1)
    bp_output = model.bilinear_pool(features_flatten)
    output = model.fc(bp_output)
    
 
    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output, 1).item()
    pred_class = output[:, pred]
 
    # 求中间变量features的梯度
    # 方法1
    # features.register_hook(extract)
    # pred_class.backward()
    # 方法2
    features_grad = autograd.grad(pred_class, features, allow_unused=True)[0]
 
    grads = features_grad  # 获取梯度
    # 使用全局平局池化（GAP）  本来是 h,w,d 现在降到1,1,d
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]#128,1,1
    features = features[0]#128,224,224
    print("pooled_grads:", pooled_grads.shape)
    print("features:", features.shape)
    pdb.set_trace()
    # features.shape[0]是指定层feature的通道数
    #将卷积层各通道特征值乘以梯度（每一个通道乘以一个梯度），表示每一个点位对于模型最后分类的重要性，即产生类激活图
    for i in range(features.shape[0]):  #features[i, ...] :(224,224)
        features[i, ...] *= pooled_grads[i, ...]
    pdb.set_trace()
    # 计算heatmap
    heatmap = features.detach().cpu().numpy()
    #沿着通道方向计算均值
    heatmap = np.mean(heatmap, axis=0)
    # 去除所有负数
    heatmap = np.maximum(heatmap, 0)
    # 归一化
    heatmap /= np.max(heatmap)
 
    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
 
    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.7 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
 
 
if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=448),
        torchvision.transforms.CenterCrop(size=448),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # 构建模型并加载预训练参数
    
    cot2 = BCNN(n_classes=200, pretrained=False).cuda()
    state_dict = torch.load("/home/zhuyuan/zy_all/weblyFG-dataset-main/model/plm_web-bird_bcnn_best-epoch_76.48.pth")



    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v
    cot2.load_state_dict(new_state_dict)
    # cot2.to(device)
    print(cot2)

    draw_CAM(cot2, single_img_path, save_path, transform=transform, visual_heatmap=True, out_layer=out_layer_name)




# https://blog.csdn.net/PanYHHH/article/details/113407903