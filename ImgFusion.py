import os
import cv2
import numpy as np
import DenseSIFT

def getimage(path, h, w):
    imagenamelist = os.listdir(path)
    print(imagenamelist)
    imagelist = []
    imageRGBlist = []
    N= len(imagenamelist)
    for i in range(N):
        image = cv2.imread(path+'/'+imagenamelist[i], 0)
        image = cv2.resize(image, (w, h))
        imageRGB = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        imagelist.append(image)
        imageRGBlist.append(imageRGB)
    h, w = np.shape(imagelist[0])
    img = np.zeros((N, h, w))
    imgRGB = np.zeros((N, h, w, 3))
    for i in range(N):
        img[i, :, :] = imagelist[i]
    for i in range(N):
        imgRGB[i, :, :, :] = imageRGBlist[i]
    return img, imgRGB


def imgextend(img, l):
    N, h, w = np.shape(img)
    extendimg = np.zeros((N, h+2*l, w+2*l))
    extendimg[:, l:l+h, l:l+w] = img
    return extendimg


def RF_Horizontal(img, D, sigma):
    a = np.exp(-np.sqrt(2)/sigma)
    F = img
    V = a**D
    h, w = np.shape(img)
    for i in range(1, w):
        F[:, i] = F[:, i] + V[:, i]*(F[:, i - 1] - F[:, i])
    for i in range(w-2, -1, -1):
        F[:, i] = F[:, i] + V[:, i + 1]*(F[:, i + 1] - F[:, i])
    return F


def image_transpose(img):
    T = img.T
    return T


def RF(img, sigma_s, sigma_r, joint_image, num_iterations=3):
    if np.shape(img)[0] != np.shape(joint_image)[0] or np.shape(img)[1] != np.shape(joint_image)[1]:
        print('RFerror')
    else:
        J = joint_image
        h, w, numchannels = np.shape(J)
        dIcdx = np.diff(J, axis=1)
        dIcdy = np.diff(J, axis=0)
        dIdx = np.zeros((h, w))
        dIdy = np.zeros((h, w))
        for i in range(numchannels):
            dIdx[:, 1:] = dIdx[:, 1:] + np.abs(dIcdx[:, :, i])
            dIdy[1:, :] = dIdy[1:, :] + np.abs(dIcdy[:, :, i])
        dHdx = 1 + sigma_s / sigma_r * dIdx
        dVdy = 1 + sigma_s / sigma_r * dIdy
        dVdy = dVdy.T
        F = img.copy()
        sigma_H = sigma_s
        for i in range(num_iterations-1):
            sigma_H_i = sigma_H * np.sqrt(3) * 2**(num_iterations - (i + 1)) / np.sqrt(4**num_iterations - 1)
            F = RF_Horizontal(F, dHdx, sigma_H_i)
            F = image_transpose(F)
            F = RF_Horizontal(F, dVdy, sigma_H_i)
            F = image_transpose(F)
        F = F.astype(type(img))
        return F


def dSiftFusion(img, imgRGB, scale, weighted_average='winner_take_all'):
    N, h, w = np.shape(img)
    img1 = img[:, 1:h, 1:w]
    imgRGB1 = imgRGB[:, 1:h, 1:w, :]
    dsifts = np.zeros((N, h-1, w-1, 32))
    extendimg = imgextend(img, int(scale/2-1))
    for i in range(N):
        extractor = DenseSIFT.DsiftExtractor(gridSpacing=1, patchSize=scale, nrml_thres=1.0, sigma_edge=1.0, sift_thres=0.2)
        feaArr, positions = extractor.process_image(extendimg[i, :, :], )
        # print(np.shape(feaArr), np.shape(positions))
        dsifts[i, :, :, :] = feaArr.reshape((h - 1, w - 1, 32))
    print('DenseSIFT finished!')
    contrastmap = np.zeros((N, h-1, w-1))
    for i in range(N):
        contrastmap[i, :, :] = np.sum(dsifts[i, :, :, :], axis=2)
    if weighted_average == 'winner_take_all':
        labels = np.argmax(contrastmap, axis=0)
        print('labels', labels)
        for i in range(N):
            mono = np.zeros((h-1, w-1))
            mono[np.where(labels == i)] = 1
            contrastmap[i, :, :] = mono
    exposuremap = np.ones((N, h-1, w-1))
    # exposuremap[(img1 >= 0.90) | (img1 <= 0.10)] = 0
    exposuremap[np.where(img1 <= 0.1*255)] = 0
    Tmap = exposuremap+10**(-10)
    Tmap = Tmap/np.tile(np.sum(Tmap, axis=0), (N, 1, 1))
    weightmap = contrastmap*Tmap
    print('weightmap finished!')
    for i in range(N):
        weightmap[i, :, :] = RF(weightmap[i, :, :], 100, 4, imgRGB1[i, :, :, :], 3)
    weightmap = weightmap + 10**(-10)
    weightmap = weightmap / np.tile(np.sum(weightmap, axis=0), (N, 1, 1))
    F = np.zeros((h-1, w-1, 3))
    w = np.zeros((h-1, w-1, 3))
    for i in range(N):
        for c in range(3):
            # w = np.tile(weightmap[i, :, :], (3, 1, 1))
            w[:, :, c] = weightmap[i, :, :]
        F = F+imgRGB1[i, :, :, :]*w
    F = np.uint8(F)
    return F


# path = '../img'
path = 'D:\image'
img, imgRGB = getimage(path, 600, 900)

'''winner_take_all或weighted_average两种方法'''
fusionImage = dSiftFusion(img, imgRGB, 48, weighted_average='weighted_average')
# fusionImage = cv2.resize(fusionImage, (5472, 3648))
cv2.imwrite(path + '/fusion.jpg', fusionImage)
