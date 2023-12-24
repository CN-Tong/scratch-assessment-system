import subprocess

from flask import Flask, render_template
from flask_cors import CORS  # 导入CORS模块

from ImgAcquisition import ImgAcquisition
from ImgFusion import ImgFusion
from ImgEnhancement import ImgEnhancement

# 运行---python -m http.server---启动前端服务

app = Flask(__name__)
CORS(app)  # 启用CORS，允许所有来源访问

currentTime = 0
path = ''


@app.route('/scratch/acquire', methods=['POST'])
def acquireImage():
    print('start image acquisition')

    imgAcquisition = ImgAcquisition()
    # 曝光时间1000ms
    currentTime = imgAcquisition.acquire(1000)
    path = f'./results/{currentTime}/original'
    # path = f'./results/1703128491.525947'

    imgFusion = ImgFusion()
    imgFusion.fuse(path)

    imgEnhancement = ImgEnhancement()
    imgEnhancement.enhance(path)

    return path


@app.route('/scratch/detect', methods=['POST'])
def detectScratch():
    print('start scratch detection')
    try:
        subprocess.run(['python', 'WeakScratchDetection.py'], check=True)
        return "Image acquisition successful"
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"


app.run(port=5000, debug=True)
