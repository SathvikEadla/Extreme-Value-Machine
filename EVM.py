import numpy as np
import libmr
import sys
import scipy.spatial.distance
import sklearn.metrics.pairwise
import time
from contextlib import contextmanager
from multiprocessing import Pool,cpu_count
import itertools as it
import pandas as pd
import config

@contextmanager
def timer(message):
    """
    Simple timing method. Logging should be used instead for large scale experiments.
    """
    print(message)
    start = time.time()
    yield
    stop = time.time()
    print("...elapsed time: {}".format(stop-start))


def euclidean_cdist(X,Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="euclidean", n_jobs=1)
def euclidean_pdist(X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="euclidean", n_jobs=1)
def cosine_cdist(X,Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="cosine", n_jobs=1)
def cosine_pdist(X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="cosine", n_jobs=1)

dist_func_lookup = {
    "cosine":{"cdist":cosine_cdist,
              "pdist":cosine_pdist},
    
    "euclidean":{"cdist":euclidean_cdist,
                 "pdist":euclidean_pdist}
}



cdist_func = dist_func_lookup["euclidean"]["cdist"]
pdist_func = dist_func_lookup["euclidean"]["pdist"]


def set_cover_greedy(universe,subsets,cost=lambda x:1.0):
    """
    A greedy approximation to Set Cover.
    """
    universe = set(universe)
    subsets = list(map(set,subsets))
    covered = set()
    cover_indices = []
    while covered != universe:
        max_index = (np.array([len(x - covered) for x in subsets])).argmax()
        covered |= subsets[max_index]
        cover_indices.append(max_index)
    return cover_indices

def set_cover(points,weibulls,solver=set_cover_greedy):
    """
    Generic wrapper for set cover. Takes a solver function.
    Could do a Linear Programming approximation, but the
    default greedy method is bounded in polynomial time.
    """
    universe = list(range(len(points)))
    d_mat = pdist_func(points)
    p = Pool(cpu_count())
    probs = np.array(p.map(weibull_eval_parallel,list(zip(d_mat,weibulls))))
    p.close()
    p.join()
    thresholded = list(zip(*np.where(probs >= cover_threshold)))
    subsets = {k:tuple(set(x[1] for x in v)) for k,v in it.groupby(thresholded, key=lambda x:x[0])}
    subsets = [subsets[i] for i in universe]
    keep_indices = solver(universe,subsets)
    return keep_indices

def reduce_model(points,weibulls,labels,labels_to_reduce=None):
    """
    Model reduction routine. Calls off to set cover.
    """
    if cover_threshold >= 1.0:
        # optimize for the trivial case
        return points,weibulls,labels
    ulabels = np.unique(labels)
    if labels_to_reduce == None:
        labels_to_reduce = ulabels
    labels_to_reduce = set(labels_to_reduce)
    keep = np.array([],dtype=int)
    for ulabel in ulabels:
        ind = np.where(labels == ulabel)
        if ulabel in labels_to_reduce: 
            print(("...reducing model for label {}".format(ulabel)))
            keep_ind = set_cover(points[ind],[weibulls[i] for i in ind[0]])
            keep = np.concatenate((keep,ind[0][keep_ind]))
        else:
            keep = np.concatenate((keep,ind[0]))
    points = points[keep]
    weibulls = [weibulls[i] for i in keep]
    labels = labels[keep]
    return points,weibulls,labels

def weibull_fit_parallel(args):
    """Parallelized for efficiency"""
    global tailsize
    dists,row,labels = args
    nearest = np.partition(dists[np.where(labels != labels[row])],tailsize)
    mr = libmr.MR()
    mr.fit_low(nearest,tailsize)
    return str(mr)

def weibull_eval_parallel(args):
    """Parallelized for efficiency"""
    dists,weibull_params = args
    mr = libmr.load_from_string(weibull_params)
    probs = mr.w_score_vector(dists)
    return probs

def fuse_prob_for_label(prob_mat,num_to_fuse):
    """
    Fuse over num_to_fuse extreme vectors to obtain
    probability of sample inclusion (PSI)
    """
    return np.average(np.partition(prob_mat,-num_to_fuse,axis=0)[-num_to_fuse:,:],axis=0)

def fit(X,y):
    """
    Analogous to scikit-learn\'s fit method.
    """
    global margin_scale
    d_mat = margin_scale*pdist_func(X)
    p = Pool(cpu_count())
    row_range = list(range(len(d_mat)))
    args = list(zip(d_mat,row_range,[y for i in row_range]))
    with timer("...getting weibulls"):
        weibulls = p.map(weibull_fit_parallel, args)
    p.close()
    p.join()
    return weibulls

def predict(X,points,weibulls,labels):
    """
    Analogous to scikit-learn's predict method
    except takes a few more arguments which
    constitute the actual model.
    """
    global num_to_fuse,ot
    d_mat = cdist_func(points,X).astype(np.float64)
    p = Pool(cpu_count())
    probs = np.array(p.map(weibull_eval_parallel,list(zip(d_mat,weibulls))))
    p.close()
    p.join()
    ulabels = np.unique(labels)
    fused_probs = []
    for ulabel in ulabels:
        fused_probs.append(fuse_prob_for_label(probs[np.where(labels == ulabel)],num_to_fuse))
    fused_probs = np.array(fused_probs)
    max_ind = np.argmax(fused_probs,axis=0)
    predicted_labels = ulabels[max_ind]
    confidence = fused_probs[max_ind]
    for i in range(confidence.shape[0]):
          if(confidence[i,i] < ot):
            predicted_labels[i] = 99  #if probability threshold is less than the specified value then it ia labelled as 99 value
    return predicted_labels,fused_probs


def load_data(fname):
    df = pd.read_csv(fname,header = None)
    labels = df.iloc[:,0]
    data = df.iloc[:,1:]
    return np.array(data),np.array(labels)

def get_accuracy(predictions,labels):
    return sum(predictions == labels)/float(len(predictions))

def update_params(n_tailsize,
                  n_cover_threshold,
                  n_cdist_func,
                  n_pdist_func,
                  n_num_to_fuse,
                  n_margin_scale):
    global tailsize,cover_threshold,cdist_func,pdist_func,num_to_fuse,margin_scale
    tailsize = n_tailsize
    cover_threshold = n_cover_threshold
    cdist_func = n_cdist_func
    pdist_func = n_pdist_func
    num_to_fuse = n_num_to_fuse
    margin_scale= n_margin_scale


def open_set_evm(train_fname,test_fname):

    with timer("...loading train data"):
       Xtrain,ytrain = load_data(train_fname)  
       print(Xtrain.shape,ytrain.shape)
    with timer("...loading test data"):
        Xtest, ytest = load_data(test_fname)
        print(Xtest.shape,ytest.shape)          
    with timer("...fitting train set"):
        weibulls = []
        weibulls = fit(Xtrain,ytrain)
    with timer("...reducing model"):
        Xtrain,weibulls,ytrain = reduce_model(Xtrain,weibulls,ytrain)
    print(("...model size: {}".format(len(ytrain))))
    with timer("...getting predictions"):
        predictions,probs = predict(Xtest,Xtrain,weibulls,ytrain)
    with timer("...evaluating predictions"):
        accuracy = get_accuracy(predictions,ytest)       
    print("accuracy: {}".format(accuracy))
    return accuracy,predictions,ytest
    
accuracy, predictions, yactual = open_set_evm('train.csv','test.csv')

