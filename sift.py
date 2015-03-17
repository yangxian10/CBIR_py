__author__ = '86057940_yx'
__date__ = '2015-2-9'

#from PIL import Image
import os
import numpy as np
#from pylab import *
import cv2
import cPickle

def process_image(imagename, resultname, mask=False):#params='--edge-thresh 10 --peak-thresh 5'):
    '''
    if imagename[-3:] != 'pgm':
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    cmmd = str('sift ' + imagename + ' --output=' + resultname + " " + params)
    os.chdir(os.getcwd())
    os.system(cmmd)
    '''
    img = cv2.imread(imagename)
#    img = cv2.resize(img, (300, 400))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kwargs = {
        'nfeatures': 0,
        'nOctaveLayers': 3,
        'contrastThreshold': 0.04,
        'edgeThreshold': 10,
        'sigma': 1.6
    }
    sift = cv2.SIFT(**kwargs)
    if mask:
        try:
            imagename_mask = imagename[:-4] + '_RCC' + '.png'#imagename[-4:]
            img_mask = cv2.imread(imagename_mask)
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)

            img_mask = cv2.resize(img_mask, (img.shape[1], img.shape[0]))
            keypoints, descriptors = sift.detectAndCompute(img, img_mask)
        except:
            keypoints, descriptors = sift.detectAndCompute(img, None)
    else:
        keypoints, descriptors = sift.detectAndCompute(img, None)
    write_features_to_file(resultname, keypoints, descriptors)
    print 'processed', imagename, 'to', resultname

def read_features_from_file(filename):
    '''
    f = open(filename, 'rb')
    locs, desc = cPickle.load(f)
    f.close()
    return locs, desc
    '''
    try:
        f = np.loadtxt(filename)
        return f[:,:4], f[:,4:]
    except:
        return np.zeros((1,4)), np.zeros((1,128))

def write_features_to_file(filename, locs, desc):
    '''
    f = open(filename, 'wb')
    cPickle.dump((locs,desc), f)
    f.close()
    '''
    f1 = open(filename, 'w')
    for i in range(len(locs)):
        x = locs[i].pt[0]
        y = locs[i].pt[1]
        size = locs[i].size
        angle = locs[i].angle
        f1.write('%s %s %s %s ' % (x, y, size, angle))
        f1.write(' '.join(str(v) for v in desc[i]))
        '''
        for j in range(128):
            f1.write(' %s ' % desc[i][j])
        '''
        f1.write('\n')
    f1.close()

def plot_features(im, locs, circle=False):
    keys = []
    for loc in locs:
        keys.append(cv2.KeyPoint(loc[0], loc[1], loc[2], loc[3]))
    img = cv2.drawKeypoints(im, keys, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('sift', img)
    cv2.waitKey()
    '''
    def draw_circle(c, r):
        t = np.arange(0, 1.01, .01)*2*np.pi
        x = r*np.cos(t) + c[0]
        y = r*np.sin(t) + c[1]
        plot(x, y, 'b', linewidth=2)

    imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plot(locs[:,0], locs[:,1], 'ob')
    axis('off')
    '''

if __name__ == '__main__':
    import cbir_utils
    import cv2
    '''
    path = 'F:\\img_set_shaidan\\test\\'
    imlist, featlist = cbir_utils.create_imglist_featlist(path)
    nbr_images = len(imlist)
    for i in range(nbr_images):
        process_image(imlist[i], featlist[i])

        im = Image.open(imlist[i]).convert('L')
        l1, d1 = read_features_from_file(featlist[i])
        figure()
        gray()
        plot_features(im, l1, circle=True)
        show()
        key = input()
    '''


#########################################################
    ''''''
    imgname = '2.jpg'
#    imgname = 'F:\\sn_online_3c_test\\000000000104199788_3.jpg'
#    im = Image.open(imgname).convert('L')
    im = cv2.imread(imgname)
#    im = cv2.resize(im, (300,400))
    process_image(imgname, 'lena.sift', mask = True)
    l1, d1 = read_features_from_file('lena.sift')

#    figure()
#    gray()
    plot_features(im, l1, circle=True)
#    show()
