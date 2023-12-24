import subprocess

from flask import Flask, render_template
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
# backendPath = ''
# frontendPath = ''
# 测试
backendPath = f'./static/pages/results/1703128491.525947'
frontendPath = f'./results/1703128491.525947'


@app.route('/scratch/acquire', methods=['POST'])
def acquireImage():
    print('start image acquisition')

    # imgAcquisition = ImgAcquisition()
    # # 曝光时间1000ms
    # currentTime = imgAcquisition.acquire(1000)
    # backendPath = f'./static/pages/results/{currentTime}'
    # frontendPath = f'./pages/results/{currentTime}'

    imgFusion = ImgFusion()
    imgFusion.fuse(backendPath)

    imgEnhancement = ImgEnhancement()
    imgEnhancement.enhance(backendPath)

    return frontendPath + '/fusion.jpg'


@app.route('/scratch/detect', methods=['POST'])
def detectScratch():
    print('start scratch detection')

    T1, T2, T3, T4 = 5, 1, 60, 5
    scratchWide = 3
    theta, d, g = 10, 3, 20
    detection = Detection(backendPath + '/eh.jpg', T1, T2, T3, T4)
    detection.Coarse_Detection(backendPath + '/coarse.jpg', scratchWide)
    detection.Fine_Detection(backendPath + '/coarse.jpg', backendPath + '/eh.jpg', backendPath, theta, d, g)

    return frontendPath + '/result.jpg'


@app.route('/scratch/evaluate', methods=['POST'])
def evaluate():
    print('start evaluate')

    evaluate = Evaluate()
    result = evaluate.evaluate(backendPath)

    return result


app.run(port=5000, debug=True)
