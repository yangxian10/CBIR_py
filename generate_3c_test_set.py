__author__ = '86057940_yx'
__date__ = '2015-2-13'

import cv2
import os
import re

if __name__ == '__main__':
    path = 'F:\\sn_online_3c\\'
    out_path = 'F:\\sn_online_3c_test\\'
    subsample = 1000

    sub_path_set = os.listdir(path)
    for sub_path in sub_path_set:
        filename_set = [(file) for file in os.listdir(path + sub_path) if re.match('.+\.jpg$', file)]
        for filename in filename_set[::subsample]:
            img = cv2.imread(path + sub_path + '\\' + filename)
            cv2.imwrite(out_path+ '\\' + filename, img)