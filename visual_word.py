__author__ = '86057940_yx'
__date__ = '2015-2-10'

import scipy.cluster.vq as cluster
import numpy as np
import sift
from sklearn.cluster import MiniBatchKMeans

class Vocabulary(object):
    def __init__(self, name):
        self.name = name
        self.voc = []
        self.idf = []
        self.traindata = []
        self.nbr_word = 0

    def train(self, featurefiles, k=100, subsampling=10):
        nbr_images = len(featurefiles)
        descr = []
        descr.append(sift.read_features_from_file(featurefiles[0])[1])
        descriptors = descr[0]
        print "begin loading image feature files..."
        for i in np.arange(1, nbr_images):
            descr.append(sift.read_features_from_file(featurefiles[i])[1])
#                descriptors = np.vstack((descriptors, descr[i]))
            descriptors = np.vstack((descriptors, descr[i][::subsampling,:]))
            if i%100 == 0:
                print i, "images have been loaded..."
        print "finish loading image feature files!"

#        self.voc, distortion = cluster.kmeans(descriptors[::subsampling,:], k, 1)
        print "begin MiniBatchKMeans cluster....patient"
        mbk = MiniBatchKMeans(k, init="k-means++", compute_labels=False, n_init=3, init_size=3*k)
#        mbk.fit(descriptors[::subsampling,:])
        mbk.fit(descriptors)
        self.voc = mbk.cluster_centers_
        print "cluster finish!"
        self.nbr_word = self.voc.shape[0]
        imwords = np.zeros((nbr_images, self.nbr_word))
        for i in xrange(nbr_images):
            imwords[i] = self.project(descr[i])

        nbr_occurences = np.sum((imwords > 0)*1, axis=0)
        self.idf = np.log( (1.0*nbr_images) / (1.0*nbr_occurences+1) )
        self.traindata = featurefiles

    def project(self, descriptors):
        imhist = np.zeros((self.nbr_word))
        words, distance = cluster.vq(descriptors, self.voc)
        for i in words:
            imhist[i] += 1

        return imhist

    def get_words(self, descriptors):
        return vq(descriptors, self.voc)[0]

if __name__ == '__main__':
    import pickle
    import cbir_utils

    path = 'F:\\img_set_shaidan\\test\\'
    imlist, featlist = cbir_utils.create_imglist_featlist(path)

    voc = Vocabulary('su_imgtest')
    voc.train(featlist, 1000, 10)

    with open('su_imgtest.pkl', 'wb') as f:
        pickle.dump(voc, f)
    print 'vocabulary is', voc.name, voc.nbr_word