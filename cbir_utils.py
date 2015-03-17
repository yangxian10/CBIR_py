__author__ = '86057940_yx'
__date__ = '2015-2-11'

import os
import re

def create_imglist_featlist(path):
    file_names = os.listdir(path)
    imglist = [os.path.join(path,filename) for filename in file_names if re.match('.+\.jpg$', filename)]
#    featlist = [filename for filename in file_names if re.match('.+\.sift$', filename)]
    featlist = [imgname[:-3]+'sift' for imgname in imglist]
    return imglist, featlist

if __name__ == '__main__':
    imglist, featlist = create_imglist_featlist('F:\\img_set_shaidan\\test\\')
    print imglist
    print featlist