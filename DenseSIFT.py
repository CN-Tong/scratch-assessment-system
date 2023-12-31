import numpy as np
from scipy import signal
from matplotlib import pyplot

# sift features
Nangles = 8
Nbins = 2
Nsamples = Nbins ** 2
alpha = 9.0
angles = np.array(range(Nangles)) * 2.0 * np.pi / Nangles


def gen_dgauss(sigma):
    '''
    generating a derivative of Gauss filter on both the X and Y
    direction.//在X和Y方向上生成高斯滤波器的导数。
    '''
    fwid = int(2 * np.ceil(sigma))
    G = np.array(range(-fwid, fwid + 1)) ** 2
    G = G.reshape((G.size, 1)) + G
    G = np.exp(- G / 2.0 / sigma / sigma)
    G /= np.sum(G)
    GH, GW = np.gradient(G)
    GH *= 2.0 / np.sum(np.abs(GH))
    GW *= 2.0 / np.sum(np.abs(GW))
    return GH, GW


class DsiftExtractor:
    '''
    The class that does dense sift feature extractor.//进行密集筛选的类提取器
    Sample Usage:
        extractor = DsiftExtractor(gridSpacing,patchSize,[optional params])
        feaArr,positions = extractor.process_image(Image)
    '''

    def __init__(self, gridSpacing, patchSize,
                 nrml_thres=1.0, \
                 sigma_edge=0.8, \
                 sift_thres=0.2):
        '''
        gridSpacing: the spacing for sampling dense descriptors//密集描述符的采样间隔
        patchSize: the size for each sift patch//每个sift patch的尺寸
        nrml_thres: low contrast normalization threshold//低对比度归一化阈值
        sigma_edge: the standard deviation for the gaussian smoothing//高斯平滑的标准差
            before computing the gradient
        sift_thres: sift thresholding (0.2 works well based on
            Lowe's SIFT paper)//sift阈值化(0.2基于Lowe's sift paper效果很好)
        '''
        self.gS = gridSpacing
        self.pS = patchSize
        self.nrml_thres = nrml_thres
        self.sigma = sigma_edge
        self.sift_thres = sift_thres
        # compute the weight contribution map
        sample_res = self.pS / np.double(Nbins)
        sample_p = np.array(range(self.pS))
        sample_ph, sample_pw = np.meshgrid(sample_p, sample_p)
        sample_ph.resize(sample_ph.size)
        sample_pw.resize(sample_pw.size)
        bincenter = np.array(range(1, Nbins * 2, 2)) / 2.0 / Nbins * self.pS - 0.5
        bincenter_h, bincenter_w = np.meshgrid(bincenter, bincenter)
        bincenter_h.resize((bincenter_h.size, 1))
        bincenter_w.resize((bincenter_w.size, 1))
        dist_ph = abs(sample_ph - bincenter_h)
        dist_pw = abs(sample_pw - bincenter_w)
        weights_h = dist_ph / sample_res
        weights_w = dist_pw / sample_res
        weights_h = (1 - weights_h) * (weights_h <= 1)
        weights_w = (1 - weights_w) * (weights_w <= 1)
        # weights is the contribution of each pixel to the corresponding bin center
        self.weights = weights_h * weights_w
        # pyplot.imshow(self.weights)
        # pyplot.show()

    def process_image(self, image, positionNormalize=True, \
                      verbose=True):
        '''
        processes a single image, return the locations
        and the values of detected SIFT features.//处理单个图像，返回检测到的SIFT特征的位置和值。
        image: a M*N image which is a numpy 2D array. If you
            pass a color image, it will automatically be converted
            to a grayscale image.//一个M*N的图像，它是一个numpy二维数组。如果您传递一个彩色图像，它将自动转换为灰度图像。
        positionNormalize: whether to normalize the positions
            to [0,1]. If False, the pixel-based positions of the
            top-right position of the patches is returned.//是否将位置规范化为[0,1]。如果为False，则返回补丁右上角的基于像素的位置。

        Return values:
        feaArr: the feature array, each row is a feature//特征数组，每一行都是一个特征
        positions: the positions of the features//特征的位置
        '''

        image = image.astype(np.double)
        if image.ndim == 3:
            # we do not deal with color images.
            image = np.mean(image, axis=0)
        # compute the grids
        H, W = image.shape
        gS = self.gS
        pS = self.pS
        remH = np.mod(H - pS, gS)
        remW = np.mod(W - pS, gS)
        offsetH = remH // 2
        offsetW = remW // 2
        gridH, gridW = np.meshgrid(range(offsetH, H - pS + 1, gS), range(offsetW, W - pS + 1, gS))

        gridH = gridH.flatten()
        gridW = gridW.flatten()
        if verbose:
            print('Image: w {}, h {}, gs {}, ps {}, nFea {}'. \
                  format(W, H, gS, pS, gridH.size))
        feaArr = self.calculate_sift_grid(image, gridH, gridW)
        # feaArr = self.normalize_sift(feaArr)
        if positionNormalize:
            positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        else:
            positions = np.vstack((gridH, gridW))
        return feaArr, positions

    def calculate_sift_grid(self, image, gridH, gridW):
        '''
        This function calculates the unnormalized sift features
        It is called by process_image().//此函数计算未规范化的sift特性。
                                        //它被process_image()调用。
        '''
        H, W = image.shape
        Npatches = gridH.size
        feaArr = np.zeros((Npatches, Nsamples * Nangles))

        # calculate gradient
        GH, GW = gen_dgauss(self.sigma)
        IH = signal.convolve2d(image, GH, mode='same')
        IW = signal.convolve2d(image, GW, mode='same')
        Imag = np.sqrt(IH ** 2 + IW ** 2)
        Itheta = np.arctan2(IH, IW)
        Iorient = np.zeros((Nangles, H, W))
        for i in range(Nangles):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - angles[i]) ** alpha, 0)
            # pyplot.imshow(Iorient[i])
            # pyplot.show()
        for i in range(Npatches):
            currFeature = np.zeros((Nangles, Nsamples))
            for j in range(Nangles):
                currFeature[j] = np.dot(self.weights, \
                                        Iorient[j, gridH[i]:gridH[i] + self.pS, gridW[i]:gridW[i] + self.pS].flatten())
            feaArr[i] = currFeature.flatten()
        return feaArr

    def normalize_sift(self, feaArr):
        '''
        This function does sift feature normalization
        following David Lowe's definition (normalize length ->
        thresholding at 0.2 -> renormalize length)
        '''
        siftlen = np.sqrt(np.sum(feaArr ** 2, axis=1))
        hcontrast = (siftlen >= self.nrml_thres)
        siftlen[siftlen < self.nrml_thres] = self.nrml_thres
        # normalize with contrast thresholding
        feaArr /= siftlen.reshape((siftlen.size, 1))
        # suppress large gradients
        feaArr[feaArr > self.sift_thres] = self.sift_thres
        # renormalize high-contrast ones
        feaArr[hcontrast] /= np.sqrt(np.sum(feaArr[hcontrast] ** 2, axis=1)). \
            reshape((feaArr[hcontrast].shape[0], 1))
        return feaArr


class SingleSiftExtractor(DsiftExtractor):
    '''
    The simple wrapper class that does feature extraction, treating
    the whole image as a local image patch.//一个简单的封装类，它能把整个图像当作一个局部图像补丁
    '''

    def __init__(self, patchSize,
                 nrml_thres=1.0, \
                 sigma_edge=0.8, \
                 sift_thres=0.2):
        # simply call the super class __init__ with a large gridSpace
        DsiftExtractor.__init__(self, patchSize, patchSize, nrml_thres, sigma_edge, sift_thres)

    def process_image(self, image):
        return DsiftExtractor.process_image(self, image, False, False)[0]


if __name__ == '__main__':
    # ignore this. I only use this for testing purpose...
    from scipy import misc
    import imageio

    extractor = DsiftExtractor(8, 16, 1)
    image = imageio.imread('C:/Users/qgl/Desktop/articles/test1.png')
    image = np.mean(np.double(image), axis=2)
    feaArr, positions = extractor.process_image(image)
    # pyplot.hist(feaArr.flatten(),bins=100)
    # pyplot.imshow(feaArr[:256])
    # pyplot.plot(np.sum(feaArr,axis=0))
    pyplot.imshow(feaArr[np.random.permutation(feaArr.shape[0])[:256]])

    # test single sift extractor
    extractor = SingleSiftExtractor(16)
    feaArrSingle = extractor.process_image(image[:16, :16])
    pyplot.figure()
    pyplot.plot(feaArr[0], 'r')
    pyplot.plot(feaArrSingle, 'b')
    pyplot.show()