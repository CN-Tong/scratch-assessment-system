import subprocess

from flask import Flask, render_template
from flask import request
from flask_cors import CORS  # 导入CORS模块

from Evaluate import Evaluate
from ImgAcquisition import ImgAcquisition
from ImgFusion import ImgFusion
from ImgEnhancement import ImgEnhancement
from WeakScratchDetection import Detection

# 运行---python -m http.server---启动前端服务

app = Flask(__name__)
CORS(app)  # 启用CORS，允许所有来源访问

currentTime = 0
# 生产
# backendPath = ''
# frontendPath = ''

# 测试
# backendPath = f'./static/pages/results/1703128491.525947'
# frontendPath = f'./results/1703128491.525947'


@app.route('/scratch/acquire', methods=['POST'])
def acquireImage():
    print('start image acquisition')


    imgAcquisition = ImgAcquisition()
    # 曝光时间1000ms
    currentTime = imgAcquisition.acquire(1000)
    backendPath = f'./static/pages/results/{currentTime}'
    frontendPath = f'/pages/results/{currentTime}'

    imgFusion = ImgFusion()
    imgFusion.fuse(backendPath)

    imgEnhancement = ImgEnhancement()
    imgEnhancement.enhance(backendPath)

    return [frontendPath, backendPath]


@app.route('/scratch/detect', methods=['POST'])
def detectScratch():
    print('start scratch detection')

    backendPath = request.json['backendPath']
    print(backendPath)

    '''T1和T4最终当硬件确定后，选择最合适的曝光参数'''
    T1, T2, T3, T4 = 45, 5, 40, 40
    '''细划痕scratchWide取3，粗划痕scratchWide取4'''
    scratchWide = 3
    theta, d, g = 10, 3, 20

    print(backendPath + 'eh.jpg')
    detection = Detection(backendPath + '/eh.jpg', T1, T2, T3, T4)
    detection.Coarse_Detection(backendPath + '/coarse.jpg', scratchWide)
    detection.Fine_Detection(backendPath + '/coarse.jpg', backendPath + '/eh.jpg', backendPath, theta, d, g)

    return "scratch detection successfully!"


@app.route('/scratch/evaluate', methods=['POST'])
def evaluate():
    print('start evaluate')

    backendPath = request.json['backendPath']
    print(backendPath)

    evaluate = Evaluate()
    result = evaluate.evaluate(backendPath)

    return result


app.run(port=5000, debug=True)
