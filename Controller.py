import os
import subprocess

from flask import Flask, render_template
from flask_cors import CORS  # 导入CORS模块

# 运行---python -m http.server---启动前端服务

app = Flask(__name__)
CORS(app)  # 启用CORS，允许所有来源访问


@app.route('/scratch/acquire', methods=['POST'])
def acquireImage():
    print('start image acquisition')
    try:
        subprocess.run(['python', 'ImgFusion.py'], check=True)
        return "Image acquisition successful"
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"


# @app.route('/scratch/enhance', methods=['POST'])
# def enhanceImage():
#     print('start image enhancement')
#     try:
#         subprocess.run(['python', 'ImgEnhancement.py'], check=True)
#         return "Image acquisition successful"
#     except subprocess.CalledProcessError as e:
#         return f"Error: {e}"


@app.route('/scratch/detect', methods=['POST'])
def detectScratch():
    print('start scratch detection')
    try:
        subprocess.run(['python', 'ImgEnhancement.py'], check=True)
        subprocess.run(['python', 'WeakScratchDetection.py'], check=True)
        return "Image acquisition successful"
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"


app.run(port=5000, debug=True)
