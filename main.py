import numpy as np
from Model import Model
from Layer import FC
import Func
import matplotlib.pyplot as plt
from math import pi
import os
from PIL import Image
import random
from Optimizer import Adam

if __name__ == "__main__":
    imgs = []
    labels = []
    imgpath = []

    for root, dirs, files in os.walk("./train_2022"):
        for name in files:
            # print(root, dirs, files[:3], name)
            img = Image.open(os.path.join(root, name))
            img = np.asarray(img, dtype='float32')
            # imgs.append(img.flatten())
            label = np.zeros(10)
            label = int(root.split('\\')[-1])
            labels.append(label)
            imgs.append(img)

    imgs = np.array(imgs)
    labels = np.array(labels)
    # 使用打乱的索引来重新排序数据集

    mymodel = Model([])
    mymodel.load_parameter("model.pkl")
    print("accuracy:{:.2f}".format(np.sum(labels == np.argmax(mymodel.predict(imgs), axis=1)) / len(labels)))