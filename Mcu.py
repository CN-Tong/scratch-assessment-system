import serial  # 导入串口通信库
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

ser = serial.Serial()

b = 0
col = False


def port_open_recv():  # 对串口的参数进行配置
    ser.baudrate = 115200
    ser.bytesize = 8
    ser.parity = 'N'
    ser.stopbits = 1
    ser.timeout = 1
    ser.port = 'COM3'
    # if ser.name()==None:
    #     return-1
    ser.open()
    if (ser.isOpen()):
        print("串口打开成功！")
        return 0
    else:
        print("串口打开失败！")
        return -1


# isOpen()函数来查看串口的开闭状态


def port_close():
    ser.close()
    if (ser.isOpen()):
        print("串口关闭失败！")
    else:
        print("串口关闭成功！")


def sendStop():
    if (ser.isOpen()):
        # [1]00停止 [1]01非采集正方向移动 [1]02[5]00非采集负方向移动 [1]02[5]01采集负方向移动
        send_data = [0xAA, 0x00, 0x30, 0x10, 0x20, 0x00, 0xcc]

        ser.write(bytes(send_data))  # 编码
        print("发送成功", send_data)
    else:
        print("发送失败！")


def sendUp():
    if (ser.isOpen()):
        # [1]00停止 [1]01非采集正方向移动 [1]02[5]00非采集负方向移动 [1]02[5]01采集负方向移动
        send_data = [0xAA, 0x01, 0x30, 0x10, 0x20, 0x00, 0xcc]

        ser.write(bytes(send_data))  # 编码
        print("发送成功", send_data)
    else:
        print("发送失败！")


def sendDownColl():
    if (ser.isOpen()):
        # [1]00停止 [1]01非采集正方向移动 [1]02[5]00非采集负方向移动 [1]02[5]01采集负方向移动
        send_data = [0xAA, 0x02, 0x30, 0x10, 0x20, 0x01, 0xcc]

        ser.write(bytes(send_data))  # 编码
        print("发送成功", send_data)
    else:
        print("发送失败！")


if __name__ == '__main__':
    port_open_recv()

    executor = ThreadPoolExecutor(max_workers=5)
    while True:
        a = input('0stop，1up, 2downCol, other quit')
        a = int(a)
        if a == 0:
            sendStop()
        elif a == 1:
            sendUp()
        elif a == 2:
            sendDownColl()
        else:
            break

        sleep(0.5)

    port_close()
