__author__ = '86057940_yx'
__date__ = '2015-2-11'

import cPickle
from pysqlite2 import dbapi2 as sqlite
import numpy as np
import xlrd
import os

class Indexer(object):
    def __init__(self, db, voc):
        if os.path.exists(db):
            os.remove(db)
        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def db_commit(self):
        self.con.commit()

    def create_tables(self):
        self.con.execute('create table imlist(filename)')
        self.con.execute('create table imwords(imid,wordid,vocname)')
        self.con.execute('create table imhistograms(imid,histogram,vocname)')
        self.con.execute('create index im_idx on imlist(filename)')
        self.con.execute('create index wordid_idx on imwords(wordid)')
        self.con.execute('create index imid_idx on imwords(imid)')
        self.con.execute('create index imidhist_idx on imhistograms(imid)')
        self.db_commit()

    def add_to_index(self,imname,descr):
        """ Take an image with feature descriptors,
            project on vocabulary and add to database. """

        if self.is_indexed(imname): return
        print 'indexing', imname

        # get the imid
        imid = self.get_id(imname)

        # get the words
        imwords = self.voc.project(descr)
        nbr_words = imwords.shape[0]

        # link each word to image
        for i in range(nbr_words):
            word = imwords[i]
            # wordid is the word number itself
            self.con.execute( \
                "insert into imwords(imid,wordid,vocname) values (?,?,?)", (imid,word,self.voc.name))

        # store word histogram for image
        # use pickle to encode NumPy arrays as strings
        self.con.execute( \
            "insert into imhistograms(imid,histogram,vocname) values (?,?,?)", \
            (imid,cPickle.dumps(imwords),self.voc.name))

    def is_indexed(self,imname):
        """ Returns True if imname has been indexed. """

        im = self.con.execute( \
            "select rowid from imlist where filename='%s'" % imname).fetchone()
        return im != None

    def get_id(self,imname):
        """ Get an entry id and add if not present. """

        cur = self.con.execute( \
            "select rowid from imlist where filename='%s'" % imname)
        res=cur.fetchone()
        if res==None:
            cur = self.con.execute( \
                "insert into imlist(filename) values ('%s')" % imname)
            return cur.lastrowid
        else:
            return res[0]

class Searcher(object):

    def __init__(self,db,voc):
        """ Initialize with the name of the database. """
        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def candidates_from_word(self,imword):
        """ Get list of images containing imword. """

        im_ids = self.con.execute( \
            "select distinct imid from imwords where wordid=%d" % imword).fetchall()
        return [i[0] for i in im_ids]

    def candidates_from_histogram(self, imwords):
        """ Get list of images with similar words. """

        # get the word ids
        words = imwords.nonzero()[0]

        # find candidates
        candidates = []
        for word in words:
            c = self.candidates_from_word(word)
            candidates += c

        # take all unique words and reverse sort on occurrence
        tmp = [(w,candidates.count(w)) for w in set(candidates)]
        tmp.sort(cmp=lambda x, y:cmp(x[1],y[1]))
        tmp.reverse()

        # return sorted list, best matches first
        return [w[0] for w in tmp]

    def get_imhistogram(self,imname):
        """ Return the word histogram for an image. """

        im_id = self.con.execute( \
            "select rowid from imlist where filename='%s'" % imname).fetchone()
        s = self.con.execute( \
            "select histogram from imhistograms where rowid='%d'" % im_id).fetchone()

        # use pickle to decode NumPy arrays from string
        return cPickle.loads(str(s[0]))

    def query(self,imname):
        """ Find a list of matching images for imname. """

        h = self.get_imhistogram(imname)
        candidates = self.candidates_from_histogram(h)

        matchscores = []
        for imid in candidates:
            # get the name
            cand_name = self.con.execute( \
                "select filename from imlist where rowid=%d" % imid).fetchone()
            cand_h = self.get_imhistogram(cand_name)
            cand_dist = np.sqrt( sum( self.voc.idf*(h - cand_h)**2 ) )
            matchscores.append( (cand_dist,imid) )

        # return a sorted list of distances and database ids
        matchscores.sort()
        return matchscores

    def get_filename(self,imid):
        """ Return the filename for an image id. """

        s = self.con.execute( \
            "select filename from imlist where rowid='%d'" % imid).fetchone()
        return s[0]

def compute_validate_score(src, imlist, file_relation):
    """ Returns the average number of correct
        images on the top four results of queries. """

    nbr_images = len(imlist)
    relation_data = xlrd.open_workbook(file_relation)
    relation_table = relation_data.sheets()[0]
    nrows = relation_table.nrows
    relation_dict = {}
    for i in xrange(1,nrows):
        row_val = relation_table.row_values(i)
        relation_dict[row_val[0]] = (row_val[2], row_val[4])

#    pos = zeros((nbr_images, 4))
    # get first four results for each image
    total_score = 0
    total_score_catentry_group = 0
    total_score_brand_id = 0
    for i in range(nbr_images):
        # compute top1 score and return average
#        pos[i] = [w[1] for w in src.query(imlist[i])[:4]]
        top_result = src.query(imlist[i])
        if len(top_result) < 1:
            continue
        elif len(top_result) < 2:
            top1_result = top_result[0]     # there is a bug, only one query result for index 87,291
        else:
            top1_result = top_result[1]
#        print top1_result, i
        top1_name = src.get_filename(top1_result[1]).split('_')[-2].split('\\')[-1]
        query_name = imlist[i].split('_')[-2].split('\\')[-1]
        if top1_name == query_name:
            total_score += 1
        if relation_dict[top1_name][0] == relation_dict[query_name][0]:
            total_score_catentry_group += 1
        if relation_dict[top1_name][1] == relation_dict[query_name][1]:
            total_score_brand_id += 1
#    score = array([ (pos[i]//4)==(i//4) for i in range(nbr_images)])*1.0
        if i%10 == 0:
            print '=====index:', i, total_score, total_score_catentry_group, total_score_brand_id, '====='
    print 'query finished!'
    return total_score*1.0 / (nbr_images), \
           total_score_catentry_group*1.0 / (nbr_images), \
           total_score_brand_id*1.0 / (nbr_images)

#from PIL import Image
#import matplotlib.pyplot as plt
import cv2

def plot_results(src, res):
    """ Show images in result list 'res'. """
    nbr_results = len(res)
    if nbr_results == 0:
        return
    show_size = 200
    img_show = np.zeros((show_size, nbr_results*show_size, 3)).astype(np.uint8)
    for i in range(nbr_results):
        imname = src.get_filename(res[i])
        img_patch = cv2.imread(imname)
        img_patch = cv2.resize(img_patch, (show_size,show_size))
        img_show[:, i*show_size:(i+1)*show_size, :] = img_patch
    cv2.imshow('query result', img_show)
    cv2.waitKey()
    '''
    plt.figure()
    nbr_results = len(res)
    for i in range(nbr_results):
        imname = src.get_filename(res[i])
        plt.subplot(1, nbr_results, i+1)
        im = np.array(Image.open(imname))
        plt.imshow(im)
        plt.axis('off')
    plt.show()
    '''

if __name__ == '__main__':
    import cPickle
    import sift
    import image_search
    from visual_word import Vocabulary
    import cbir_utils

    path = 'F:\\img_set_shaidan\\test\\'
    imlist, featlist = cbir_utils.create_imglist_featlist(path)
    nbr_images = len(imlist)



    with open('su_imgtest.pkl', 'rb') as f:
        voc = cPickle.load(f)

    ### create image index ###
    '''
    indx = image_search.Indexer('test.db', voc)
    indx.create_tables()

    for i in range(nbr_images)[:1000]:
        locs, descr = sift.read_features_from_file(featlist[i])
        indx.add_to_index(imlist[i], descr)

    indx.db_commit()

    from pysqlite2 import dbapi2 as sqlite

    con = sqlite.connect('test.db')
    print con.execute('select count (filename) from imlist').fetchone()
    print con.execute('select * from imlist').fetchone()
    '''

    ### test image index ###
    '''
    src = image_search.Searcher('test.db', voc)
    locs, descr = sift.read_features_from_file(featlist[0])
    iw = voc.project(descr)

    print 'ask using a histogram...'
    print src.candidates_from_histogram(iw)[:10]
    '''

    ### test query ###
    '''
    src = image_search.Searcher('test.db', voc)
    print 'try a query...'
    print src.query(imlist[87])[:10]
    '''

    ### test traversal query ###
    '''
    src = image_search.Searcher('test.db', voc)
    pr = image_search.compute_validate_score(src, imlist)
    print 'query num:', nbr_images, 'precision rate: ', pr*100, '%'
    '''

    ### test show query result ###
    '''    '''
    src = image_search.Searcher('test.db', voc)
    nbr_results = 6
    for i in range(nbr_images):
        res = [w[1] for w in src.query(imlist[i])[:nbr_results]]
        image_search.plot_results(src, res)