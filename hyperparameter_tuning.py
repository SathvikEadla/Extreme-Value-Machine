
"""
######################################################################################################
                     HYPERPARAMETER TUNING
######################################################################################################

"""

from EVM import load_data
from hyperopt import hp, tpe, fmin 


Xtrain,ytrain = load_data('train.csv') 
Xtest, ytest = load_data('test.csv')

space = [hp.quniform('tailsize',20,150,5), hp.quniform('cover_threshold',0.3,0.9,0.1), 
        hp.quniform('num_to_fuse',1,10,1), hp.quniform('margin_scale',0.3,0.7,0.1) ,
        hp.quniform('ot',0,0.3,0.001)]

def open_set_evm(X_train,y_train,X_test,y_test):
  
    with timer("...fitting train set"):
        weibulls = []
        weibulls = fit(X_train,y_train)
    with timer("...reducing model"):
        X_train,weibulls,y_train = reduce_model(X_train,weibulls,y_train)
    print(("...model size: {}".format(len(y_train))))
    with timer("...getting predictions"):
        predictions,probs = predict(X_test,X_train,weibulls,y_train)
    with timer("...evaluating predictions"):
        accuracy = get_accuracy(predictions,y_test)       
    print("accuracy: {}".format(accuracy))
    return accuracy


def tune_func(args):
    global tailsize,cover_threshold,num_to_fuse,margin_scale,ot
    tailsize = int(args[0])
    cover_threshold = args[1]
    num_to_fuse = int(args[2])
    margin_scale = args[3]
    ot = args[4]
    print(args)
    accuracy = open_set_evm(Xtrain,ytrain,Xtest,ytest)
    return -accuracy


best = fmin(tune_func,space, algo=tpe.suggest, max_evals=200)
print('Best Parameters obtained are: ',best)