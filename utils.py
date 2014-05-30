# Allstate Purchase Prediction Challenge
# Author: Alessandro Mariani <alzmcr@yahoo.it>
# https://www.kaggle.com/c/allstate-purchase-prediction-challenge

'''
This module cointains the data preparation and utilities
'''

from time import time
from itertools import combinations
from sklearn import preprocessing

import scipy as sp, numpy as np, pandas as pd

# Cantor Pairing
def cantor(args):
    # Cantor Pairing - recursive call if more than 1 pair
    if len(args) > 2:
        x2 = cantor(args[1:])
        x1 = args[0]
    else:
        x1, x2 = args
    return int((0.5 * (x1 + x2)*(x1 + x2 + 1) + x2))

# Groups all columns of data into combinations of [degree]
def group_data(data, degree=3, hash=hash, NAMES=None): 
    init = time()
    new_data = []; combined_names = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
        if NAMES != None:
            combined_names.append( '+'.join([NAMES[indicies[i]] for i in range(degree)]) )
    print "DONE! %.2fm" % ((time()-init)/60)
    if NAMES != None:
        return (np.array(new_data).T, combined_names)
    return np.array(new_data).T

# Return concatenated fields in a dataframe
# [1,2,3,4,5,6] => '123456'
def concat(df, columns):
    return np.array([''.join(x) for x in np.array(
        [np.array(df[col].values, dtype=str) for col in columns]).T])

# Breakfast Pirate Awesome State trick + some additions
def stateFix(encoders,df,c=['C','D','G'],verbose=False):
    # GA
    iGA = df.state == encoders['state'].transform(['GA'])[0]
    ifix = iGA&(df[c[0]]==1); df.ix[ifix,c[0]] = 2; nga1 = np.sum(ifix) #C
    ifix = iGA&(df[c[1]]==1); df.ix[ifix,c[1]] = 2; nga2 = np.sum(ifix) #D
    # FL
    iFL = df.state == encoders['state'].transform(['FL'])[0]
    ifix = iFL&(df[c[2]]<=2); df.ix[ifix,c[2]] = 3; nfl1 = np.sum(ifix) #G
    # OH
    iOH = df.state == encoders['state'].transform(['OH'])[0]
    ifix = iOH&(df[c[2]]==1); df.ix[ifix,c[2]] = 2; noh1 = np.sum(ifix) #G
    # ND
    iND = df.state == encoders['state'].transform(['ND'])[0]
    ifix = iND&(df[c[2]]!=2); df.ix[ifix,c[2]] = 2; nnd1 = np.sum(ifix) #G
    # SD
    iSD = df.state == encoders['state'].transform(['SD'])[0]
    ifix = iSD&(df[c[2]]!=2); df.ix[ifix,c[2]] = 2; nsd1 = np.sum(ifix) #G
    if verbose:
        print "Fixed state law products. GA1:%i GA2:%i FL1:%i OH1:%i ND1:%i SD1:%i" %(
            nga1, nga2, nfl1, noh1, nnd1, nsd1)

# Target variable expected value given a categorical feature
def expval(df,col,y,tfilter):
    tmp = pd.DataFrame(index=df.index)
    pb = df[tfilter][y].mean()                                              # train set mean
    tmp['cnt'] = df[col].map(df[tfilter][col].value_counts()).fillna(0)     # train set count
    tmp['csm'] = df[col].map(df[tfilter].groupby(col)[y].sum()).fillna(pb)  # train set sum
    tmp.ix[tfilter,'cnt'] -= 1                                              # reduce count for train set
    tmp.ix[tfilter,'csm'] -= df.ix[tfilter,y]                               # remove current value
    tmp['exp'] = ((tmp.csm+ pb*15) / (tmp.cnt+ 15)).fillna(pb)              # calculate mean including kn-extra 'average' samples 
    np.random.seed(1)
    tmp.ix[tfilter,'exp'] *= 1+.3*(np.random.rand(len(tmp[tfilter]))-.5) # add some random noise to the train set
    return tmp.exp

def prepare_data(shuffle=True):
    alltest = pd.read_csv('data\\test_v2.csv')
    test = alltest.set_index('customer_ID')
    alldata = pd.read_csv('data\\train.csv').set_index('customer_ID')

    # handy lists of features
    con = ['group_size','car_age','age_oldest','age_youngest','duration_previous','cost']
    cat = ['homeowner','car_value','risk_factor','married_couple','C_previous','state', 'location','shopping_pt']
    conf = ['A','B','C','D','E','F','G']; conf_f = [col+'_f' for col in conf]
    extra = []

    final_purchase = alldata[alldata.record_type == 1]          # final purchase
    data = alldata.join(final_purchase[conf], rsuffix='_f')     # creating training dataset with target features
    data = data[data.record_type == 0]                          # removing final purchase

    data['conf'] = concat(data,conf_f)                          # handy purchase plan 
    data['conf_init'] = concat(data,conf)                       # handy last quoted plan

    encoders = dict()
    data = data.append(test)

    # Fix NAs
    data['C_previous'].fillna(0, inplace=1)
    data['duration_previous'].fillna(0, inplace=1)
    data.location.fillna(-1, inplace=1);
    # Transform data to numerical data
    for col in ['car_value','risk_factor','state']:
        encoders[col] = preprocessing.LabelEncoder()
        data[col] = encoders[col].fit_transform(data[col].fillna(99))

    print 'Location substitution:',
    ## get rid of very location, given the total count from train,cv and test set
    x = data[data.shopping_pt==2].location.value_counts()
    sub = data.location.map(x).fillna(0) < 5
    data.ix[sub,'location'] = data.state[sub]; print '%.5f' % sub.mean()

    # cost per car_age; cost per person; cost per state
    data['caCost'] = 1.*data.cost / (data.car_age+1)
    data['ppCost'] = 1.*data.cost / data.group_size
    data['stCost'] = data.state.map(data.groupby('state')['cost'].mean())
    extra.extend(['caCost','ppCost','stCost'])

    # average quote cost by G values
    data['costG'] = data['G'].map(data.groupby('G')['cost'].mean())
    extra.append('costG')

    # average quote cost by G & state values
    x = data.groupby(['G','state'])['cost'].mean()
    x = x.reset_index().set_index(['G','state']); x.columns = ['costStG']   # covert to DF
    data = data.merge(x,left_on=['G','state'],right_index=True,how='left')
    extra.append('costStG')

    # two way intersactino between state, G and shopping_pt
    print "Grouping few 2-way interactions...",
    grpTrn, c2 = group_data(data[['state','G','shopping_pt']].values,2,hash,['state','G','shopping_pt'])
    for i,col in enumerate(c2):
        encoders[col] = preprocessing.LabelEncoder()
        data[col] = encoders[col].fit_transform(grpTrn[:,i])
    extra.extend(c2)

    # expected value (arithmetic average) of G by state & location
    for col in ['state','location']:
        extra.append(col+'_exp')
        data[col+'_exp'] = expval(data,col,'G_f',-data.G_f.isnull())

    # previous G
    data['prev_G'] = data.G.shift(1); extra.append('prev_G')
    data.ix[data.shopping_pt == 1,'prev_G'] = data.ix[data.shopping_pt==1,'G']

    # separating training & test data
    test = data[data.conf.isnull()]; data = data[-data.conf.isnull()]

    # SHUFFLE THE DATASET, keeping the same customers transaction in order
    if shuffle:
        print "Shuffling dataset...",
        np.random.seed(9); ids = np.unique(data.index.values)
        rands = pd.Series(np.random.random_sample(len(ids)),index=ids)
        data['rand'] = data.reset_index()['customer_ID'].map(rands).values
        data.sort(['rand','shopping_pt'],inplace=1); print "DONE!"

    # convert to int due to emtpy values in test set
    for col in conf_f: data[col] = np.array(data[col].values,dtype=np.int8)

    return data,test,con,cat,extra,conf,conf_f,encoders

