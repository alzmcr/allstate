# Allstate Purchase Prediction Challenge
# Author: Alessandro Mariani <alzmcr@yahoo.it>
# https://www.kaggle.com/c/allstate-purchase-prediction-challenge

'''
This module is for train, cross validate and make the final prediction.
'''

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import operator

from sklearn import cross_validation, ensemble
from utils import prepare_data, concat, expval, stateFix
from parallel import RandomForestsParallel
from time import time

def majority_vote(baseline,model_predictions):
    # given a baseline and a matrix of prediction (#samples x #models)
    # if will return the prediction if 1+#models/2 agree on the same product
    # otherwise will return the baseline
    prcnt = np.vstack([np.bincount(p,minlength=5) for p in model_predictions])
    prmax = np.max(prcnt,axis=1) >= (1+(len(selected)/2))
    preds = baseline+0; preds[prmax] = np.argmax(prcnt[prmax],axis=1)
    return preds

def make_ptscores(y_true,y_pred,y_base,pt,vmask):
    # measure the increase of "plan" accuracy given a prediction for the product (G)
    return [np.mean(vmask[pt==ipt]&(y_true[pt==ipt] == y_pred[pt==ipt])) - np.mean(vmask[pt==ipt]&(y_true[pt==ipt] == y_base[pt==ipt])) for ipt in range(1,11)]

if __name__ == '__main__':
    ############################################################################
    ## SETTING #################################################################
    # submit: if 'True' create a submission file and train models for submission
    # N: number of models to build
    # NS: number of models to selected for majority vote
    # kfold: number of k-fold to perform, if not submitting
    # N_proc: number of process to spawn, default #CPU(s)-1
    # include_from_pt: minimum shopping_pt included in the data set
    # verbose_selection: print all details while selecting the model
    # tn: test set distrubution of shopping_pt (#10-11 merged)    
    ############################################################################
    submit = True; N = 50; NS = 9; kfold = 3; N_proc = None;
    include_from_pt = 1; verbose_selection = False
    tn = np.array([18943,13298,9251,6528,4203,2175,959,281,78])
    ############################################################################
    # Random Forest Setting ####################################################
    # Must be a list containg a tuple with (ntree,maxfea,leafsize)
    params = [(50,5,23)]
    # ex. [(x,5,23) for x in [35,50,75]] # [(50,x,23) for x in range(4,12)]
    # anything you'd like to try, here is the place for the modifications
    ############################################################################
    
    print "Majority vote using %i models, selecting %i\n" % (N,NS)
    # initialize data
    data,test,con,cat,extra,conf,conf_f,encoders = prepare_data()
    data = data[data.shopping_pt >=include_from_pt]; print "Including from shopping_pt #%i\n" % data.shopping_pt.min(),
    # features, target, weights (not used)
    X = data[con+cat+conf+extra]; y = data['G_f'] ; w = np.ones(y.shape)
    
    vmask = reduce(operator.and_,data[conf[:-1]].values.T==data[conf_f[:-1]].values.T)
    scores,imp,ptscores = {},{},{}
    for n,m,l in params:
        t = time();
        scores[(m,l)],imp[(m,l)],ptscores[(m,l)] = [],[],[]
        col_trscores,col_cvscores = [],[]

        # initialize the ensemble of forests to run in parallel
        # class is also structured to handle single-process 
        rfs = RandomForestsParallel(N, n, m, l, N_proc)
        
        # cross validation is use to find the best parameters
        for ifold,(itr,icv) in enumerate(cross_validation.KFold(len(y),kfold,indices=False)):
            if submit:
                # just a lame way to re-using the same code for fitting & selecting when submitting :)
                itr = np.ones(y.shape,dtype=bool); icv = -itr
                print "\nHEY! CREATING SUBMISSION!\n"
            else:
                # redo expected value for the current training & cv set
                for c in [x for x in X.columns if x[-4:] == '_exp']:
                    X[c] = expval(data,c[:-4],'G_f',itr)           

            # fits the random forests at the same time
            rfs.fit(X[itr],y[itr],w[itr])

            print "predicting...",
            allpreds = rfs.predict(X)
            rftscores = []
            print "selecting models..."
            for irf in range(len(rfs.rfs)):
                # SELECTION of the best random forest, even though probably
                # is just getting rid of very unlucky seeds ...
                pG = allpreds[:,irf]; ipt2 =  data.shopping_pt > 1
                ptscore = make_ptscores(y[icv],pG[icv],data.G[icv],data.shopping_pt[icv],vmask[icv])
                tptscore = make_ptscores(y[itr],pG[itr],data.G[itr],data.shopping_pt[itr],vmask[itr])
                rftscores.append((tn.dot(tptscore[1:]),irf))
                print "%i,%i %.5f %.5f %.5f %.5f" % (
                    ifold,irf,
                    np.mean(pG[itr]==y[itr]),np.mean(vmask[itr]&(pG[itr]==y[itr])),
                    np.mean(pG[ipt2&itr]==y[ipt2&itr]),np.mean(vmask[ipt2&itr]&(pG[ipt2&itr]==y[ipt2&itr]))),
                if verbose_selection:
                    print " ".join(["%.5f" %pts for pts in ptscore]),
                    print " ".join(["%.5f" %pts for pts in tptscore]),
                print "%.2f %.2f" %(tn.dot(tptscore[1:]),tn.dot(ptscore[1:]))

            # select the best models for the majority vote
            rftscores.sort(reverse=1); selected = [x[1] for x in rftscores[:NS]]

            print "counting votes..."
            # print also the score using all the models
            pG = majority_vote(data.G,allpreds)
            ptscore = make_ptscores(y[icv],pG[icv],data.G[icv],data.shopping_pt[icv],vmask[icv])
            # ifold,a : majority vote score using all models
            print str(ifold)+",a "+" ".join(["%.5f" %pts for pts in ptscore])+" %.2f" % tn.dot(ptscore[1:])
            
            # results for selected models
            pG = majority_vote(data.G,allpreds[:,selected])
            ptscore = make_ptscores(y[icv],pG[icv],data.G[icv],data.shopping_pt[icv],vmask[icv])
            # ifold,s : majority vote score using selected models
            print str(ifold)+",s "+" ".join(["%.5f" %pts for pts in ptscore])+" %.2f" % tn.dot(ptscore[1:])
            
            # append features importances & scores
            col_trscores.append(np.mean(pG[itr]==y[itr]))        # append train score
            col_cvscores.append(np.mean(pG[icv]==y[icv]))        # append cv score
            imp[(m,l)].append(rfs.impf)
            scores[(m,l)].append(tn.dot(ptscore[1:]))
            ptscores[(m,l)].append(ptscore)

            # skip any following fold if we're submitting
            if submit: break
        
        print "%i %i %i\t %.2f %.2f %.4f %.4f %.2f - %.2fm" % (
            n,m,l,
            np.mean(scores[(m,l)]), np.std(scores[(m,l)]),  # for best params & variance
            np.mean(col_trscores), np.mean(col_cvscores),   # use x diagnostic training set overfit
            tn.dot(np.mean(ptscores[(m,l)],axis=0)[1:]),    # score
            (time()-t)/60),                                 # k-fold time
        print " ".join(["%.5f" %pts for pts in np.mean(ptscores[(m,l)],axis=0)]),
        print " ".join(["%.5f" %pts for pts in np.std(ptscores[(m,l)],axis=0)])
        
    if submit:
        # MAKE SUBMISSION
        # very complicated way to keep only the latest shopping_pt for each customer just to have everything in one row!!!!!11
        test = test[test.shopping_pt == test.reset_index().customer_ID.map(test.reset_index().groupby('customer_ID').shopping_pt.max())]
        Xt = test[con+cat+conf+extra]

        # TEST SET PREDICTION
        print "now predicting on test set...",
        allpreds = rfs.predict(Xt)
        test['pG'] = majority_vote(test.G,allpreds[:,selected]); print "done"
        
        # Fix state law products, then concatenate to string
        stateFix(encoders,test,['C','D','pG'],1)
        test['plan'] = concat(test,['A','B','C','D','E','F','pG'])
        test['plan'].to_csv('submission\\majority_rfs%i_%i.%i_shuffle_GAfix_%iof%iof%i.csv' % (
            n,m,l,NS/2+1,NS,N),header=1)

        # features importances
        impf = rfs.impf; impf.sort()
        

