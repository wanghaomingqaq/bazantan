from scipy.io import loadmat
import numpy as np
from PIL import Image
import os



def to_data(file):
    # 使用 loadmat 函数加载 MAT 文件
    print(file)
    mat_data = loadmat(file)
    data = mat_data['finalMaterialRecording']
    print(data['materialNameEnglish'])
    image = data['displayImage'][0][0]
    normalForce = data['normalForce']
    frictionForce = data['frictionForce']
    # 摩擦力
    fric_coff = frictionForce / normalForce
    mu = fric_coff[0][0]
    print(mu.shape)
    print("end")
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.show()

# 替换下面的路径为您的MAT文件路径
path = 'E:\读研\学术\触觉生成\LMT_FinalDatabase\C1\S1\M1'
for root, dirs, files in os.walk(path):
    for file in files:
        paths = root + file
        to_data(paths)
