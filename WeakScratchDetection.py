import numpy as np
import pandas
import cv2
import math
import copy
from RANSAC import *
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error


class Detection():
    def __init__(self, image_path, T1=40, T2=3, T3=40, T4=32, m=20, n=20):
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        self.T4 = T4
        self.m = m
        self.n = n
        self.src_image = cv2.imread(image_path, 0)
        self.image_info = self.src_image.shape
        self.image_height = self.image_info[0]
        self.image_width = self.image_info[1]

        # hist, bins, _ = plt.hist(self.src_image.ravel(), 256, [0, 255])
        # max_index = np.argmax(hist)
        # print(max_index)
        # self.T1 = max_index + T1
        # self.T4 = max_index + T4

        # if T1:
        #     T1_add = 0
        #     n1 = int(self.image_width / n)
        #     n2 = int(self.image_height / m)
        #     for i in range(n1):
        #         for j in range(n2):
        #             T1_add += self.src_image[m * j, n * i]
        #     T1_add /= (n1 * n2)
        #     T1_add = int(T1_add)
        #     print(T1_add)
        #     self.T1 += T1_add
        #     self.T4 += T1_add

    """Rules of coarse detection"""
    """
        粗检测中的四条规则，判断是否满足，如果不满足返回True，否则返回False

        Return：bool
    """

    def judge(self, lc, ll, lr):
        if lc > self.T1:
            if (lc > (ll + self.T2)) and (lc > (lr + self.T2)):
                if abs(int(ll) - int(lr)) < self.T3:
                    if (ll > self.T4) or (lr > self.T4):
                        return False
        return True

    # def point_to_line(self, point_x, point_y, line_x0, line_y0, line_x1, line_y1):
    #     A = (line_y0-line_y1)/(line_x1*line_y0-line_x0*line_y1)
    #     B = (line_x0-line_x1)/(line_y1*line_x0-line_y0*line_x1)
    #     d = (point_x*A+point_y*B-1)/(np.sqrt(A**2+B**2))
    #     return d

    """Determine whether to merge the same line segments"""
    """
        精检测阶段判断是否满足合并两条线段的三条规则，如果满足返回True，否则返回False

        Return：bool
    """

    def merge(self, line_1, line_2, theta=5, d=5, g=20):
        """
        根据三个原则判断line_1盒line_2是否划分为同一组直线段
        :param line_1: 直线段1
        :param line_2: 直线段2
        :param theta: 允许的最大夹角
        :param d: 允许的最大中点到对方直线距离
        :param g: 允许的最大线段间隙
        :return:
        """
        k1 = (line_1[3] - line_1[1]) / -(line_1[2] - line_1[0] + 0.0001)
        k2 = (line_2[3] - line_2[1]) / -(line_2[2] - line_2[0] + 0.0001)
        angle = math.atan((k1 - k2) / (1 + k1 * k2))
        if angle <= theta:
            line1_center_x = (line_1[0] + line_1[2]) / 2
            line1_center_y = (line_1[1] + line_1[3]) / 2
            line2_center_x = (line_2[0] + line_2[2]) / 2
            line2_center_y = (line_2[1] + line_2[3]) / 2
            A12 = (line_2[1] - line_2[3]) / (- line_2[2] * line_2[1] + line_2[0] * line_2[3])
            B12 = (line_2[2] - line_2[0]) / (- line_2[2] * line_2[1] + line_2[0] * line_2[3])
            d12 = (line1_center_x * A12 + line1_center_y * B12 - 1) / (np.sqrt(A12 ** 2 + B12 ** 2))
            A21 = (line_1[1] - line_1[3]) / (- line_1[2] * line_1[1] + line_1[0] * line_1[3])
            B21 = (line_1[2] - line_1[0]) / (- line_1[2] * line_1[1] + line_1[0] * line_1[3])
            d21 = (line2_center_x * A21 + line2_center_y * B21 - 1) / (np.sqrt(A21 ** 2 + B21 ** 2))
            if d12 <= d and d21 <= d:
                gap1 = math.sqrt((line_2[0] - line_1[0]) ** 2 + (line_2[1] - line_1[1]) ** 2)
                gap2 = math.sqrt((line_2[0] - line_1[2]) ** 2 + (line_2[1] - line_1[3]) ** 2)
                gap3 = math.sqrt((line_2[2] - line_1[0]) ** 2 + (line_2[3] - line_1[1]) ** 2)
                gap4 = math.sqrt((line_2[2] - line_1[2]) ** 2 + (line_2[3] - line_1[3]) ** 2)
                gap = min(gap1, gap2, gap3, gap4)
                if gap <= g:
                    return True
        return False

    def sort_line(self, lines):
        min_value = 10000000
        max_value = 0
        if (lines[1] < min_value):
            min_value = lines[1]
        if (lines[3] < min_value):
            min_value = lines[3]
            lines[0], lines[1], lines[2], lines[3] = lines[2], lines[3], lines[0], lines[1]
        if (lines[3] > max_value):
            max_value = lines[3]
        if (lines[1] > max_value):
            max_value = lines[1]
            lines[0], lines[1], lines[2], lines[3] = lines[2], lines[3], lines[0], lines[1]

    """Returns the maximum or minimum point in the line segment set"""
    """
        lines：检测的到的所有线段->[[x1, y1, x2, y2],....]

        Return：线段集合中最小点下标和最大点下标
    """

    def get_minmax_line(self, lines):
        min_index, min_value = 0, 10000000
        max_index, max_value = 0, 0
        for i in range(len(lines)):
            if lines[i][1] < min_value:
                min_index = i
                min_value = lines[i][1]
            if lines[i][3] < min_value:
                min_index = i
                min_value = lines[i][3]
            if lines[i][3] > max_value:
                max_index = i
                max_value = lines[i][3]
            if lines[i][1] > max_value:
                max_index = i
                max_value = lines[i][1]
        return min_index, max_index

    """Avoid overflow"""
    """
        x：数值
        low：下界
        high：上界

        Return：溢出检测后的x值
    """

    def crop(self, x, low, high):
        if (x > high):
            x = high
        elif (x < low):
            x = low
        return x

    """Coordinate affine transformation to find ROI region"""
    """
        x, y：原图中点的坐标
        angle：进行仿射变换时需要图片变换的角度，使用的是直线与水平方向夹角
        scale：仿射变换时缩放的尺度

        Return：仿射变换后的坐标
    """

    def coordinate_transform(self, x, y, angle, scale):
        x = x - self.image_height * 0.5
        y = y - self.image_width * 0.5
        aff_center_x = round(x * math.cos(angle) * scale + y * math.sin(angle) * scale + self.image_height * 0.5)
        aff_center_x = self.crop(aff_center_x, 0, self.image_height)
        aff_center_y = round(-x * math.sin(angle) * scale + y * math.cos(angle) * scale + self.image_width * 0.5)
        aff_center_y = self.crop(aff_center_y, 0, self.image_width)
        return aff_center_x, aff_center_y

    """Returns the ROI region coordinate points and discards points that are not in the region"""
    """
        min_point：一条线段集合中的最小点[x1, y1, x2, y2]
        max_point：一条线段集合中的最大点[x1, y1, x2, y2]
        point：线段集合中的所有点[[x1, y1, x2, y2],....]
        drop_index：需要丢弃的点下标，用来精检测第三步丢弃点

        Rrturn：ROI区域四个坐标点(左上, 左下, 右上, 右下)
    """

    def get_ROI_drop(self, min_point, max_point, point):
        drop_index = []
        k = (max_point[3] - min_point[1]) / (max_point[2] - min_point[0] + 0.0001)
        center_x = (max_point[2] - min_point[0]) / 2
        center_y = (max_point[3] - min_point[1]) / 2
        angle = math.atan(k)
        aff_center_x, aff_center_y = self.coordinate_transform(center_x, center_y, -angle, 0.5)
        aff_ROI_top_left_x = aff_center_x - 2.5
        aff_ROI_top_left_y = aff_center_y - 25
        aff_ROI_down_left_x = aff_center_x + 2.5
        aff_ROI_down_left_y = aff_center_y - 25
        aff_ROI_top_right_x = aff_center_x - 2.5
        aff_ROI_top_right_y = aff_center_y + 25
        aff_ROI_down_right_x = aff_center_x + 2.5
        aff_ROI_down_right_y = aff_center_y + 25
        # print(aff_ROI_top_left_x, aff_ROI_top_left_y, aff_ROI_down_left_x, aff_ROI_down_left_y, aff_ROI_top_right_x,
        #       aff_ROI_top_right_y, aff_ROI_down_right_x, aff_ROI_down_right_y)
        for i in range(len(point)):
            aff_start_point_x, aff_start_point_y = self.coordinate_transform(point[i][0], point[i][1], -angle, 0.5)
            aff_end_point_x, aff_end_point_y = self.coordinate_transform(point[i][2], point[i][3], -angle, 0.5)
            if ((aff_start_point_x < aff_ROI_top_left_x) or (aff_start_point_x > aff_ROI_down_left_x)):
                drop_index.append(i)
            elif ((aff_start_point_y < aff_ROI_top_left_y) or (aff_start_point_y > aff_ROI_top_right_y)):
                drop_index.append(i)
            if ((aff_end_point_x < aff_ROI_top_left_x) or (aff_end_point_x > aff_ROI_down_left_x)):
                drop_index.append(i)
            elif ((aff_end_point_y < aff_ROI_top_left_y) or (aff_end_point_y > aff_ROI_top_right_y)):
                drop_index.append(i)
        ROI_top_left_x, ROI_top_left_y = self.coordinate_transform(aff_ROI_top_left_x, aff_ROI_top_left_y, angle, 2)
        ROI_down_left_x, ROI_down_left_y = self.coordinate_transform(aff_ROI_down_left_x, aff_ROI_down_left_y, angle, 2)
        ROI_top_right_x, ROI_top_right_y = self.coordinate_transform(aff_ROI_top_right_x, aff_ROI_top_right_y, angle, 2)
        ROI_down_right_x, ROI_down_right_y = self.coordinate_transform(aff_ROI_down_right_x, aff_ROI_down_right_y,
                                                                       angle, 2)
        # print(ROI_top_left_x, ROI_top_left_y, ROI_down_left_x, ROI_down_left_y, ROI_top_right_x, ROI_top_right_y,
        #       ROI_down_right_x, ROI_down_right_y)
        return [ROI_top_left_x, ROI_top_left_y, ROI_down_left_x, ROI_down_left_y, ROI_top_right_x, ROI_top_right_y,
                ROI_down_right_x, ROI_down_right_y], drop_index

    """粗检测"""

    def Coarse_Detection(self, coarse_detection_name, scratchWide=3):
        print("start coarse detection")
        dst_image = numpy.zeros(self.image_info)
        for i in range(scratchWide, self.image_height - scratchWide):
            for j in range(scratchWide, self.image_width - scratchWide):
                flag = True
                Lc = self.src_image[i, j]
                Ll_0 = self.src_image[i - scratchWide, j]
                Lr_0 = self.src_image[i + scratchWide, j]
                flag = self.judge(Lc, Ll_0, Lr_0)
                if flag:
                    Ll_45 = self.src_image[i - scratchWide, j - scratchWide]
                    Lr_45 = self.src_image[i + scratchWide, j + scratchWide]
                    flag = self.judge(Lc, Ll_45, Lr_45)
                if flag:
                    Ll_90 = self.src_image[i, j - scratchWide]
                    Lr_90 = self.src_image[i, j + scratchWide]
                    flag = self.judge(Lc, Ll_90, Lr_90)
                if flag:
                    Ll_135 = self.src_image[i + scratchWide, j - scratchWide]
                    Lr_135 = self.src_image[i - scratchWide, j + scratchWide]
                    flag = self.judge(Lc, Ll_135, Lr_135)
                if flag == False:
                    dst_image[i, j] = 255
        cv2.imwrite(coarse_detection_name, dst_image)

        return

    """精检测"""

    def Fine_Detection(self, binaryImgPath, enhancementPath, fine_detection_name, theta=8, d=5, g=20):
        print("start fine detection")
        binary_image = cv2.imread(binaryImgPath, 0)
        binary_image = cv2.medianBlur(binary_image, 3)
        binary_imageRGB = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
        enhancement = cv2.imread(enhancementPath)

        """step1: LSD检测并合并线段，获得同类线段集合"""
        print("step 1")
        ls = cv2.ximgproc.createFastLineDetector()
        line_result = ls.detect(binary_image)
        line_result = line_result.reshape((-1, 4))
        line_result = line_result.tolist()
        # line_result嵌套关系为[[x0, y0, x1, y1],...]
        print('line_result_num:', len(line_result))
        temp1 = binary_imageRGB.copy()
        for dline in line_result:
            x0 = int(dline[0])
            y0 = int(dline[1])
            x1 = int(dline[2])
            y1 = int(dline[3])
            cv2.line(temp1, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv2.imwrite(f'{fine_detection_name}/fld.jpg', temp1)
        # 去除长度小于阈值的直线段
        # for line_point in line_result:
        #     line_length = np.sqrt((line_point[0]-line_point[2])**2+(line_point[1]-line_point[3])**2)
        #     if line_length < 30:
        #         line_result.remove([line_point[0], line_point[1], line_point[2], line_point[3]])
        # print('line_result_num_new:', len(line_result))
        # 根据三个标准将检测到的直线段分组
        index = numpy.zeros((len(line_result)))
        merge_result, group_result = [], []
        q = 1
        while True:
            for i in range(len(line_result)):
                if index[i] == 0:
                    group_result.append(line_result[i])
                    index[i] = q
                    break
            while True:
                end_flag = True
                for i in range(len(line_result)):
                    for j in range(len(group_result)):
                        flag = self.merge(group_result[j], line_result[i], theta, d, g)
                        if flag and index[i] == 0:
                            index[i] = q
                            group_result.append(line_result[i])
                            end_flag = False
                if end_flag:
                    merge_result.append(copy.deepcopy(group_result))
                    group_result.clear()
                    q += 1
                    break
            if np.min(index) != 0:
                break

        # 去除长度小于阈值的直线段
        for merge_points in merge_result:
            for merge_point in merge_points:
                line_length = int(np.sqrt((merge_point[0] - merge_point[2]) ** 2
                                          + (merge_point[1] - merge_point[3]) ** 2))
                if line_length <= 20 and len(merge_points) <= 6:
                    merge_points.remove(merge_point)
        print('merge_result_group_num:', len(merge_result))
        # merge_result嵌套关系为[[[x0, y0, x1, y1], ...], ...]
        # merge = np.zeros(np.shape(binary_imageRGB))
        merge = binary_imageRGB.copy()
        temp2 = np.zeros(enhancement.shape)
        for group in merge_result:
            for line_in_group in group:
                x0 = int(line_in_group[0])
                y0 = int(line_in_group[1])
                x1 = int(line_in_group[2])
                y1 = int(line_in_group[3])
                cv2.line(merge, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(temp2, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(f'{fine_detection_name}/binaryResult.jpg', merge)
        cv2.imwrite(f'{fine_detection_name}/forEvaluateResult.jpg', temp2)

        """step2：根据第一步的结果设置ROI区域并将区域外点和线段丢弃"""
        # print("step 2")
        # roi_results, drop_indexs = [], []
        # for i in range(len(merge_result)):
        #     min_index, max_index = self.get_minmax_line(merge_result[i])
        #     line = copy.deepcopy(merge_result[i])
        #     roi_result, drop_index = self.get_ROI_drop(merge_result[i][min_index], merge_result[i][max_index], line)
        #     roi_results.append(roi_result)
        #     drop_indexs.append(copy.deepcopy(drop_index))
        # print('ROI_results_num:', len(roi_results))
        # print('drop_indexs:', drop_indexs)
        # for i in range(len(merge_result)):
        #     for j in range(len(drop_indexs[i])):
        #         merge_result[i].pop(drop_indexs[i][j])

        """step3：根据第二步结果，使用RANSAC拟合直线，将最相关的点找出"""
        # print("step 3")
        # points = []
        # for i in range(len(merge_result)):
        #     point = []
        #     for j in range(len(merge_result[i])):
        #         point.append(numpy.array([merge_result[i][j][0], merge_result[i][j][1]]))
        #         point.append(numpy.array([merge_result[i][j][2], merge_result[i][j][3]]))
        #     points.append(numpy.array(point))
        # # points = numpy.array(points)
        # n_input, n_output = 1, 1
        # input_columns, output_columns = range(n_input), [n_input + i for i in range(n_output)]
        # model = LinearLeastSquareModel(input_columns, output_columns, debug=False)
        # RANSAC_data = []
        # temp = binary_imageRGB.copy()
        # for i in range(len(points)):
        #     n = int(len(points) * 0.4)+1
        #     d = int(len(points) * 0.7)
        #     if len(points[i]) < 10:
        #         continue
        #     ransac_fit, ransac_data = ransac(points[i], model, n, 1000, 7e3, d, debug=False, return_all=True)
        #     data = numpy.array(ransac_data['inliers'])
        #     for j in range(data.shape[0] - 1):
        #         cv2.line(temp, (int(points[i][data[j]][0]), int(points[i][data[j]][1])),
        #                  (int(points[i][data[j + 1]][0]), int(points[i][data[j + 1]][1])), (255, 0, 0))
        # cv2.imwrite(f'{fine_detection_name}_Ransac.png', temp)

        """step4：根据第三步结果，对相关点得到的二值化图像进行概率霍夫变换进行直线检测"""
        # print("step 4")
        # image = cv2.imread(f'{fine_detection_name}_RANSAC.bmp', 0)
        # cv2.HoughLinesP(image, 1, numpy.pi / 180, 160, minLineLength=10, maxLineGap=100)
        # cv2.imwrite(f'{fine_detection_name}_Hough.bmp', image)

        """step5：多项式拟合"""
        # print("step 5")
        # temp3 = enhancement.copy()
        # for merge_points in merge_result:
        #     x = []
        #     y = []
        #     for merge_point in merge_points:
        #         x.append(int(merge_point[0]))
        #         y.append(int(merge_point[1]))
        #         x.append(int(merge_point[2]))
        #         y.append(int(merge_point[3]))
        #         xlength = merge_point[2] - merge_point[0]
        #         ylength = merge_point[3] - merge_point[1]
        #         quantity = int((max(abs(xlength), abs(ylength)))/10)
        #         for point in range(quantity):
        #             weight = random.random()
        #             x.append(int(merge_point[0] + xlength * weight))
        #             y.append(int(merge_point[1] + ylength * weight))
        #     if len(x):
        #         xMax = max(x)
        #         xMin = min(x)
        #         x = np.array(x)
        #         y = np.array(y)
        #
        #         fit = np.polyfit(x, y, 15)
        #         function = np.poly1d(fit)
        #
        #         for i in range(xMin, xMax+1):
        #             j = int(np.polyval(function, i))
        #             cv2.circle(temp3, (i, j), 1, (255, 0, 0), -1)
        #
        # cv2.imwrite(f'{fine_detection_name}/fitResult.jpg', temp3)

        '''step6:Ransac多项式拟合'''
        # print("step 6")
        # temp4 = enhancement.copy()
        # for merge_points in merge_result:
        #     if not merge_points:
        #         continue
        #     x = []
        #     y = []
        #     for merge_point in merge_points:
        #         if not merge_point:
        #             continue
        #         x.append(int(merge_point[0]))
        #         y.append(int(merge_point[1]))
        #         x.append(int(merge_point[2]))
        #         y.append(int(merge_point[3]))
        #         xlength = merge_point[2] - merge_point[0]
        #         ylength = merge_point[3] - merge_point[1]
        #         quantity = int((max(abs(xlength), abs(ylength))) / 10)
        #         for point in range(quantity):
        #             weight = random.random()
        #             x.append(int(merge_point[0] + xlength * weight))
        #             y.append(int(merge_point[1] + ylength * weight))
        #
        #     initial = int(0.3 * len(x))
        #     end = int(0.7 * len(x))
        #     fit_x, fit_y = self.fit_curve_ransac(x, y, initial, end, 100, 10, 5)
        #
        #     for i in range(fit_x):
        #         cv2.circle(temp4, (fit_x[i], fit_y[i]), 1, (0, 0, 255), -1)
        #
        # cv2.imwrite(f'{fine_detection_name}_fitRansac.bmp', temp4)

    def fit_curve_ransac(self, x, y, initial, end, k, threshold, deg):
        best_model = None
        best_inliers = None
        best_error = np.inf

        for i in range(k):
            # 随机选择n个点来拟合直线
            random_indices = np.random.choice(len(x), size=initial, replace=False)
            x_samples = []
            y_samples = []
            for i in random_indices:
                x_samples.append(x[i])
                y_samples.append(y[i])

            # 拟合直线模型
            try:
                model = np.polyfit(x_samples, y_samples, deg)
                predicted_y = np.polyval(model, x)
            except:
                continue

            # 计算拟合误差
            error = np.abs(predicted_y - y)

            # 根据误差阈值找出内点
            inliers = np.where(error < threshold)[0]

            # 更新最佳模型
            try:
                if len(inliers) > len(best_inliers) and len(inliers) > end:
                    best_model = model
                    best_inliers = inliers
                    best_error = mean_squared_error(y[best_inliers], predicted_y[best_inliers])
            except:
                continue

        fit_y = []
        for xi in x:
            fit_y.append(int(np.polyval(best_model, xi)))

        return x, fit_y

# path = '../img'
# path = 'D:\image'
# srcName = '/eh'
# srcType = '.jpg'
# T1, T2, T3, T4 = 5, 1, 60, 5
# scratchWide = 3
# theta, d, g = 10, 3, 20
# detection = Detection(path + srcName + srcType, T1, T2, T3, T4)
# detection.Coarse_Detection(path + srcName + 'Coarse.jpg', scratchWide)
# detection.Fine_Detection(path + srcName + 'Coarse.jpg', path + srcName + srcType, path + srcName, theta, d, g)
