from collections import Counter, defaultdict
from itertools import product
import os
import pickle
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
from mord import LogisticAT
from step2_fit_model_delirium import MyCalibrator, LTRPairwise, MyLogisticRegression


if __name__=='__main__':
    model_type = 'ltr'
    random_state = 2020
    suffix = '_not_combine_slowing_Aaron'
    
    ## load model
    
    with open(f'results_{model_type}_Nbt0{suffix}.pickle', 'rb') as ff:
        res = pickle.load(ff)
    for k in res:
        exec('%s = res[\'%s\']'%(k,k))
    
    ## load dataset
    with pd.ExcelFile(f'data_to_fit{suffix}.xlsx') as xls:#
        dfX = pd.read_excel(xls, 'X')
        dfy = pd.read_excel(xls, 'y')
        df_info = pd.read_excel(xls, 'info')
        df_worst_delirium_names = pd.read_excel(xls, 'worst_delirium_names')
    worst_delirium_Xnames = df_worst_delirium_names.EEGName.values.astype(str)
    dfy = dfy.rename(columns={
            'Deceased at hosp disch (0=N; 1=Y)':'DeceasedDisch',
            'Deceased at 3-mo post disch.  (0=N, 1=Y, 2=Unk)':'Deceased3month',
            'Disch. GOS':'GOSDisch'})
            
    ## for duplicate patients, pick initial visit
    
    counter = Counter(df_info.MRN)
    duplicate_mrns = [x for x in counter if counter[x]>1]
    exclude_ids = []
    for mrn in duplicate_mrns:
        ids = np.where(df_info.MRN==mrn)[0]
        init_date = df_info['Eval. date/time'].iloc[ids].min()
        exclude_ids.extend(ids[df_info['Eval. date/time'].iloc[ids]!=init_date])
    ids = ~np.in1d(np.arange(len(df_info)), exclude_ids)
    dfX = dfX[ids].reset_index(drop=True)
    dfy = dfy[ids].reset_index(drop=True)
    df_info = df_info[ids].reset_index(drop=True)

    ## get CAM-S-LF
    
    dfy.loc[dfy['CAM-S LF (0-19)']>=15, 'CAM-S LF (0-19)'] = 15
    CAMSLF = dfy['CAM-S LF (0-19)'].values
    
    assert len(set(dfX.MRN))==len(dfX)

    ## get predicted VE-CAM-S
    
    X = dfX[Xnames].values.astype(float)
    worst_delirium_mask = np.in1d(Xnames, worst_delirium_Xnames)
    good_ids = np.all(X[:,worst_delirium_mask]!=1, axis=1)
    X = X[good_ids][:,~worst_delirium_mask]
    VECAMS = np.zeros(len(dfX))+20
    VECAMS[good_ids] = model.base_estimator.predict_z(X)
    
    ## get data
    
    families = ['binomial', 'binomial', 'binomial']
    
    Xnames = ['Age', 'Sex', 'Score']
    X_CAMSLF = np.c_[df_info.Age.values, (df_info.Gender=='M').astype(float), CAMSLF]
    X_VECAMS = np.c_[df_info.Age.values, (df_info.Gender=='M').astype(float), VECAMS]
    ynames = ['DeceasedDisch', 'Deceased3month', 'GOSDisch']
    ys = dfy[ynames].values
    ys[:,-1] = (ys[:,-1]<=3).astype(int) # make GOS binary
    
    ## get CV splits
    
    Ncv = 5
    cv_te_ids = np.arange(len(ys))
    np.random.seed(random_state)
    np.random.shuffle(cv_te_ids)
    cv_te_ids = np.array_split(cv_te_ids, Ncv)
    
    ## bootstrap
    Nbt = 1000
    n_jobs = 5
    params = {}
    Xstds = {}
    coefs = defaultdict(list)
    cv_score_tes = defaultdict(list)
    cfs = defaultdict(list)
    for bti in tqdm(range(Nbt+1)):
        if bti==0:
            btids = np.arange(len(ys))
            X_CAMSLF_bt = np.array(X_CAMSLF)
            X_VECAMS_bt = np.array(X_VECAMS)
            ys_bt = np.array(ys)
        else:
            btids = np.random.choice(len(ys), len(ys), replace=True)
            X_CAMSLF_bt = X_CAMSLF[btids]
            X_VECAMS_bt = X_VECAMS[btids]
            ys_bt = ys[btids]
        
        for X_type, y_type in product(['CAMSLF','VECAMS'], ynames):
            X = eval(f'X_{X_type}_bt')
            y = ys_bt[:,ynames.index(y_type)]
            family = families[ynames.index(y_type)]
            
            cv_score_te = []
            cf = 0
            for cvi in range(Ncv):
                teids = np.in1d(btids, cv_te_ids[cvi])
                Xtr = X[~teids]
                ytr = y[~teids]
                Xte = X[teids]
                yte = y[teids]
                
                if y_type=='Deceased3month':
                    Xtr = Xtr[np.in1d(ytr, [0,1])]
                    ytr = ytr[np.in1d(ytr, [0,1])]
                    Xte = Xte[np.in1d(yte, [0,1])]
                    yte = yte[np.in1d(yte, [0,1])]
                
                # normalize
                Xmean = Xtr.mean(axis=0)
                Xstd = Xtr.std(axis=0)
                Xtr = (Xtr-Xmean)/Xstd
                Xte = (Xte-Xmean)/Xstd
                
                # define model
                if family=='binomial':
                    model = LogisticRegression(penalty='l2', class_weight='balanced', random_state=random_state+1, max_iter=1000)
                    model_params = {'C':[0.1,1,10,100]}
                    metric = 'balanced_accuracy'
                elif family=='ordinal':
                    model = LogisticAT(max_iter=1000)
                    model_params = {'alpha':[0.01,0.1,1,10]}
                    metric = 'balanced_accuracy'
                else:
                    raise NotImplementedError(family)
                
                # fit model
                if params.get((X_type,y_type, cvi)) is None:
                    model = GridSearchCV(
                        model, model_params,
                        scoring=metric,
                        n_jobs=n_jobs, refit=True, cv=Ncv)
                    model.fit(Xtr, ytr)
                    params[(X_type,y_type, cvi)] = model.best_params_
                    model = model.best_estimator_
                else:
                    for k,v in params[(X_type,y_type, cvi)].items():
                        setattr(model, k, v)
                    model.fit(Xtr, ytr)
                
                # TODO calibrate (does not affect accuracy)
                
                # predict on testing set
                ypte_prob = model.predict_proba(Xte)
                ypte = model.predict(Xte)
                
                # get testing performance
                if family=='binomial':
                    tp = np.sum((yte==1)&(ypte==1))
                    tn = np.sum((yte==0)&(ypte==0))
                    fp = np.sum((yte==0)&(ypte==1))
                    fn = np.sum((yte==1)&(ypte==0))
                    perf_metric = pd.DataFrame(
                    data={'val':[
                        roc_auc_score(yte, ypte_prob[:,1]),
                        average_precision_score(yte, ypte_prob[:,1]),
                        accuracy_score(yte, ypte),
                        fp/(tn+fp),
                        fn/(tp+fn),
                        tp/(tp+fp),
                        tn/(tn+fn),
                        cohen_kappa_score(yte, ypte),
                        f1_score(yte, ypte, average='weighted'),]},
                    index=[
                        'AUROC',
                        'AUPRC',
                        'accuracy',
                        'FPR',
                        'FNR',
                        'PPV',
                        'NPV',
                        'Cohen\'s kappa',
                        'weighted f1 score',])
                else:
                    raise NotImplementedError(family)
                #TODO
                #elif family=='ordinal':
                    
                cv_score_te.append(perf_metric)
                cf += confusion_matrix(yte, ypte)
            
            # refit to get final model and coefficients
            if family=='binomial':
                model = LogisticRegression(penalty='l2', class_weight='balanced', random_state=random_state+1, max_iter=1000)
            elif family=='ordinal':
                model = LogisticAT(max_iter=1000)
            
            param_names = model_params.keys()
            param_folds = {}
            for p in param_names:
                cc = Counter([params.get((X_type,y_type, cvi))[p] for cvi in range(Ncv)])
                most_common_count = cc.most_common(1)[0][1]
                most_common_value = [k for k,v in cc.items() if v==most_common_count]
                # takes the strongest penalty if tie
                if type(model).__name__=='LogisticRegression':
                    param_folds[p] = min(most_common_value)
                elif type(model).__name__=='LogisticAT':
                    param_folds[p] = max(most_common_value)
            for k,v in param_folds.items():
                setattr(model, k, v)
                
            if y_type=='Deceased3month':
                X = X[np.in1d(y, [0,1])]
                y = y[np.in1d(y, [0,1])]
            # normalize
            Xmean = X.mean(axis=0)
            Xstd = X.std(axis=0)
            X = (X-Xmean)/Xstd
            if bti==0:
                Xstds[(X_type, y_type)] = Xstd
            
            model.fit(X, y)
            
            coefs[(X_type, y_type)].append(model.coef_.flatten())
            cv_score_tes[(X_type,y_type)].append(sum(cv_score_te)/len(cv_score_te))
            cfs[(X_type,y_type)].append(cf)
    
    # save results
    with pd.ExcelWriter(f'outcome_coefs{suffix}.xlsx') as writer:
        for k, v in coefs.items():
            vs = np.array(v)
            vs = vs/Xstds[k]
            lb, ub = np.percentile(vs[1:], (2.5,97.5), axis=0)
            OR = np.exp(vs)
            OR_lb, OR_ub = np.percentile(OR[1:], (2.5,97.5), axis=0)
            pval = np.mean(vs[1:]>=0, axis=0)
            pval = np.minimum(pval, 1-pval)*2
            df_coef = pd.DataFrame(data={
                'feature':Xnames,
                'coef':vs[0],
                'lb':lb,
                'ub':ub,
                'OR':OR[0],
                'OR_lb':OR_lb,
                'OR_ub':OR_ub,
                'p':pval,
                })
            df_coef['coef'] = df_coef.coef.apply(lambda x: f'{x:.2f}') + df_coef.lb.apply(lambda x: f' ({x:.2f} -- ') + df_coef.ub.apply(lambda x: f' {x:.2f})')
            df_coef['OR'] = df_coef.OR.apply(lambda x: f'{x:.2f}') + df_coef.OR_lb.apply(lambda x: f' ({x:.2f} -- ') + df_coef.OR_ub.apply(lambda x: f' {x:.2f})')
            df_coef = df_coef.drop(columns=['lb','ub','OR_lb','OR_ub'])
            df_coef.to_excel(writer, sheet_name=str(k), index=False)
            
    with pd.ExcelWriter(f'outcome_model_performances{suffix}.xlsx') as writer:
        for k, v in cv_score_tes.items():
            vs = np.array([x.values.flatten() for x in v[1:]])
            lb, ub = np.percentile(vs, (2.5,97.5), axis=0)
            v[0]['lb'] = lb
            v[0]['ub'] = ub
            print('\n'+str(k))
            print(v[0])
            print(cfs[k][0])
            v[0].to_excel(writer, sheet_name=str(k))
    
    with open(f'figures/outcome_model_data_for_boxplot{suffix}.pickle','wb') as ff:
        pickle.dump(cv_score_tes, ff)
