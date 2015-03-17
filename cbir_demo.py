__author__ = '86057940_yx'
__date__ = '2015-2-13'

import cbir_utils
import visual_word
import image_search
import sift
from visual_word import Vocabulary
import cv2
import numpy as np

import cPickle

def cbir_train(train_path, voc_name, db_name, n_subsample=2000, n_cluster=2000, subfeatsampling=10):
    voc_name = voc_name + '_' + str(n_subsample) + '_' + str(n_cluster) + '_' + str(subfeatsampling)
    db_name = db_name[:-3] + '_' + str(n_subsample) + '_' + str(n_cluster) + '_' + str(subfeatsampling) + db_name[-3:]

    imlist, featlist = cbir_utils.create_imglist_featlist(train_path)
    imlist = imlist[:n_subsample]
    featlist = featlist[:n_subsample]

    ### generate sift feature
    nbr_images = len(imlist)
    ''''''
    for i in range(nbr_images):
        sift.process_image(imlist[i], featlist[i], mask = True)

    ### generate visual word
    voc = visual_word.Vocabulary(voc_name)
    voc.train(featlist, n_cluster, subfeatsampling)
    with open(voc_name+'.pkl', 'wb') as f:
        cPickle.dump(voc, f)
    print 'vocabulary is', voc.name, voc.nbr_word

    ### generate image index
    with open(voc_name+'.pkl', 'rb') as f:
        voc = cPickle.load(f)

    indx = image_search.Indexer(db_name, voc)
    indx.create_tables()

    for i in range(nbr_images):
        locs, descr = sift.read_features_from_file(featlist[i])
        indx.add_to_index(imlist[i], descr)

    indx.db_commit()
    print 'generate index finish!'
    print 'training over'

def cbir_test(test_path, voc_name, db_name, file_relation, \
              flag_show=True, nbr_results=6, \
              n_subsample=2000, n_cluster=2000, subfeatsampling=5):
    voc_name = voc_name + '_' + str(n_subsample) + '_' + str(n_cluster) + '_' + str(subfeatsampling)
    db_name = db_name[:-3] + '_' + str(n_subsample) + '_' + str(n_cluster) + '_' + str(subfeatsampling) + db_name[-3:]
    imlist, featlist = cbir_utils.create_imglist_featlist(test_path)
    imlist = imlist[:n_subsample:5]
#    imlist = imlist[2000:]
    nbr_images = len(imlist)
    with open(voc_name+'.pkl', 'rb') as f:
        voc = cPickle.load(f)
    src = image_search.Searcher(db_name, voc)

    if flag_show:
        show_size = 200
        for i in range(nbr_images):
            res = [w[1] for w in src.query(imlist[i])[:nbr_results]]
            imgnameshow = [imgname for imgname in imlist \
                   if imgname.split('_')[-2].split('\\')[-1] \
                   == imlist[i].split('_')[-2].split('\\')[-1]]
            nbr_results2 = len(imgnameshow)
            img_show = np.zeros((show_size, nbr_results2*show_size, 3)).astype(np.uint8)
            for j in range(nbr_results2):
                img_patch = cv2.imread(imgnameshow[j])
                img_patch = cv2.resize(img_patch, (show_size,show_size))
                img_show[:, j*show_size:(j+1)*show_size, :] = img_patch
            cv2.imshow('subset', img_show)
            image_search.plot_results(src, res)
    else:
        pr, pr_catentry_group, pr_brand_id = image_search.compute_validate_score(src, imlist, file_relation)
        print 'query num:', nbr_images, 'precision rate: ', pr*100, '%'
        print 'query num:', nbr_images, 'catentry group precision rate: ', pr_catentry_group*100, '%'
        print 'query num:', nbr_images, 'brand id precision rate: ', pr_brand_id*100, '%'



if __name__ == '__main__':
#    path = 'F:\\img_set_shaidan\\test\\'
#    voc_name = 'su_imgtest'
#    db_name = 'test.db'

#    path = 'F:\\sn_online_3c_test\\'
#    voc_name = 'su_online_3c_test'
#    db_name = 'su_online_3c_test.db'

#    train_path = 'F:\\img_set_shaidan\\itsm_small\\'
#    test_path = 'F:\\img_set_shaidan\\itsm_test\\'
    train_path = 'F:\\img_set_shaidan\\Saliency\\'
    test_path = 'F:\\img_set_shaidan\\Saliency\\'
    voc_name = 'su_online_shaidan_small_mbk_saliency'
    db_name = 'su_online_shaidan_small_mbk_saliency.db'
    file_relation = "shaidan_relation.xlsx"


    cbir_train(train_path, voc_name, db_name, \
               n_subsample=4000, n_cluster=500, subfeatsampling=2)
    cbir_test(train_path, voc_name, db_name, file_relation, flag_show=False, \
              n_subsample=4000, n_cluster=500, subfeatsampling=2)