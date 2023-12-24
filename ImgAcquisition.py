# coding=utf-8
import time

import mvsdk
import serial  # 导入串口通信库
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed


class ImgAcquisition:

    ser = serial.Serial()

    def acquire(self, exposureTime):
        # 枚举相机
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        for i, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
        i = 0 if nDev == 1 else int(input("Select camera: "))
        DevInfo = DevList[i]
        print(DevInfo)

        # 打开相机
        hCamera = 0
        try:
            hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(hCamera)
        ImgAcquisition.PrintCapbility(self, cap)

        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(hCamera, 0)

        # 手动曝光，曝光时间 exposureTime ms
        mvsdk.CameraSetAeState(hCamera, 0)
        mvsdk.CameraSetExposureTime(hCamera, exposureTime * 1000)

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(hCamera)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
        pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

        # 打开串口
        ImgAcquisition.port_open_recv(self)

        currentTime = time.time()
        for i in range(5):
            ImgAcquisition.sendDownColl(self)
            sleep(1)
            # 从相机取一帧图片
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 2000)
                mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

                # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
                # 该示例中我们只是把图片保存到硬盘文件中
                status = mvsdk.CameraSaveImage(hCamera, f"./static/pages/results/{currentTime}/original/{i}.bmp",
                                               pFrameBuffer, FrameHead, mvsdk.FILE_BMP,100)
                if status == mvsdk.CAMERA_STATUS_SUCCESS:
                    print("Save image successfully. image_size = {}X{}".format(FrameHead.iWidth, FrameHead.iHeight))
                else:
                    print("Save image failed. err={}".format(status))
            except mvsdk.CameraException as e:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
        ImgAcquisition.sendUp(self)

        # 关闭相机
        mvsdk.CameraUnInit(hCamera)

        # 释放帧缓存
        mvsdk.CameraAlignFree(pFrameBuffer)

        # 关闭串口
        ImgAcquisition.port_close(self)

        return currentTime


    def PrintCapbility(self, cap):
        for i in range(cap.iTriggerDesc):
            desc = cap.pTriggerDesc[i]
            print("{}: {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iImageSizeDesc):
            desc = cap.pImageSizeDesc[i]
            print("{}: {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iClrTempDesc):
            desc = cap.pClrTempDesc[i]
            print("{}: {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iMediaTypeDesc):
            desc = cap.pMediaTypeDesc[i]
            print("{}: {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iFrameSpeedDesc):
            desc = cap.pFrameSpeedDesc[i]
            print("{}: {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iPackLenDesc):
            desc = cap.pPackLenDesc[i]
            print("{}: {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iPresetLut):
            desc = cap.pPresetLutDesc[i]
            print("{}: {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iAeAlmSwDesc):
            desc = cap.pAeAlmSwDesc[i]
            print("{}: {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iAeAlmHdDesc):
            desc = cap.pAeAlmHdDesc[i]
            print("{}: {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iBayerDecAlmSwDesc):
            desc = cap.pBayerDecAlmSwDesc[i]
            print("{}: {}".format(desc.iIndex, desc.GetDescription()))
        for i in range(cap.iBayerDecAlmHdDesc):
            desc = cap.pBayerDecAlmHdDesc[i]
            print("{}: {}".format(desc.iIndex, desc.GetDescription()))





    def port_open_recv(self):  # 对串口的参数进行配置
        ImgAcquisition.ser.baudrate = 115200
        ImgAcquisition.ser.bytesize = 8
        ImgAcquisition.ser.parity = 'N'
        ImgAcquisition.ser.stopbits = 1
        ImgAcquisition.ser.timeout = 1
        ImgAcquisition.ser.port = 'COM3'
        # if ImgAcquisition.ser.name()==None:
        #     return-1
        ImgAcquisition.ser.open()
        if (ImgAcquisition.ser.isOpen()):
            print("串口打开成功！")
            return 0
        else:
            print("串口打开失败！")
            return -1


    # isOpen()函数来查看串口的开闭状态


    def port_close(self):
        ImgAcquisition.ser.close()
        if (ImgAcquisition.ser.isOpen()):
            print("串口关闭失败！")
        else:
            print("串口关闭成功！")


    def sendStop(self):
        if (ImgAcquisition.ser.isOpen()):
            # [1]00停止 [1]01非采集正方向移动 [1]02[5]00非采集负方向移动 [1]02[5]01采集负方向移动
            send_data = [0xAA, 0x00, 0x64, 0x00, 0x01, 0x00, 0xcc]

            ImgAcquisition.ser.write(bytes(send_data))  # 编码
            print("发送成功", send_data)
        else:
            print("发送失败！")


    def sendUp(self):
        if (ImgAcquisition.ser.isOpen()):
            # [1]00停止 [1]01非采集正方向移动 [1]02[5]00非采集负方向移动 [1]02[5]01采集负方向移动
            send_data = [0xAA, 0x01, 0x64, 0x00, 0x01, 0x00, 0xcc]

            ImgAcquisition.ser.write(bytes(send_data))  # 编码
            print("发送成功", send_data)
        else:
            print("发送失败！")


    def sendDownColl(self):
        if (ImgAcquisition.ser.isOpen()):
            # [1]00停止 [1]01非采集正方向移动 [1]02[5]00非采集负方向移动 [1]02[5]01采集负方向移动
            send_data = [0xAA, 0x02, 0x64, 0x00, 0x01, 0x01, 0xcc]

            ImgAcquisition.ser.write(bytes(send_data))  # 编码
            print("发送成功", send_data)
        else:
            print("发送失败！")


    def sendReturn(self):
        if (ImgAcquisition.ser.isOpen()):
            # [1]00停止 [1]01非采集正方向移动 [1]02[5]00非采集负方向移动 [1]02[5]01采集负方向移动
            send_data = [0xAA, 0x01, 0x30, 0x10, 0x20, 0x01, 0xcc]

            ImgAcquisition.ser.write(bytes(send_data))  # 编码
            print("发送成功", send_data)
        else:
            print("发送失败！")

