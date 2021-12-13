import copy
from collections import Counter, defaultdict
from itertools import combinations
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._logistic import _logistic_loss_and_grad
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from mord import LogisticAT
from tqdm import tqdm


class MyCalibrator:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        
    def fit(self, X, y):
        yp = self.predict(X)
        self.recalibration_mapper = LogisticAT(alpha=0).fit(yp.reshape(-1,1), y)
        return self
    
    def predict(self, X, z=None):
        K = len(self.base_estimator.classes_)
        if z is None:
            yp = np.sum(self.base_estimator.predict_proba(X)*np.arange(K), axis=1)
        else:
            yp = np.sum(self.base_estimator.predict_proba(X,z=z)*np.arange(K), axis=1)
        return yp
        
    def predict_proba(self, X, z=None):
        yp = self.predict(X, z=z)
        yp2 = self.recalibration_mapper.predict_proba(yp.reshape(-1,1))
        return yp2


class MyLogisticRegression(LogisticRegression):
    """
    Coefficient to some decimal (integer or 0.1x or 0.5x)
    Allows bounds
    Allow orders
    Allow coef sum constraint
    No intercept
    Binary only
    """
    def __init__(self, class_weight=None, tol=1e-6,
            C=1.0, l1_ratio=0., coef_decimal=0.1, coef_sum=None,
            random_state=None, max_iter=1000, decimal_max_iter=10000,
            bounds=None, orders=None):
        super().__init__(penalty='elasticnet', dual=False, tol=tol, C=C,
                 fit_intercept=False, intercept_scaling=1, class_weight=class_weight,
                 random_state=random_state, max_iter=max_iter,
                 multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=l1_ratio)
        self.coef_decimal = coef_decimal
        self.coef_sum = coef_sum
        self.bounds = bounds
        self.orders = orders
        self.decimal_max_iter = decimal_max_iter
                 
    def fit(self, X, y, sample_weight=None):
        if self.bounds is not None and self.orders is not None:
            for x in self.orders:
                self.bounds[x[1]] = (0, None)
                
        self.label_encoder = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder.classes_
        y = self.label_encoder.transform(y)
        
        def func(w, X, y, alpha, l1_ratio, sw):
            out, grad = _logistic_loss_and_grad(w, X, y, 0, sw)
            out_penalty = 0.5*alpha*(1 - l1_ratio)*np.sum(w**2) + alpha*l1_ratio*np.sum(np.abs(w))#[:-1]
            grad_penalty = alpha*(1-l1_ratio)*w+alpha*l1_ratio*np.sign(w)#[:-1]
            return out+out_penalty, grad+grad_penalty
        
        X2 = np.array(X)
        sum_weights = np.ones(X.shape[1])
        if self.orders is not None:
            for x in self.orders:
                X2[:,x[0]] += X2[:,x[1]]
                sum_weights[x[0]] += 1
        if self.coef_sum is not None:
            cons = ({ 'type': 'eq',
                'fun': lambda x: np.sum(x*sum_weights)-self.coef_sum,
                'jac': lambda x: sum_weights, })
        else:
            cons = None
        y2 = np.array(y)
        y2[y2==0] = -1

        #if self.bounds is None:
        #    method = 'BFGS'
        #else:
        #    method = 'L-BFGS-B'
        method = None

        if sample_weight is None:
            if self.class_weight is not None:
                sample_weight = get_sample_weights(y, class_weight=self.class_weight)
            else:
                sample_weight = np.ones(len(X))
        #sample_weight /= (np.mean(sample_weight)*len(X))

        if self.coef_sum is None:
            alpha = 1./self.C
        else:
            alpha = 1.
        w0 = np.random.randn(X.shape[1])/10
        self.opt_res = minimize(
            func, w0, method=method, jac=True,
            args=(X2, y2, alpha, self.l1_ratio, sample_weight),
            bounds=self.bounds,#+[(None,None)],
            constraints=cons,
            tol=self.tol,
            options={"maxiter": self.max_iter}
        )
        coef_ = self.opt_res.x#[:-1]
        #intercept_ = self.opt_res.x[-1]
                
        # convert coef_ to integer (up to 10^x times)
        if np.sum(np.abs(coef_)>1e-5)==0:
            coef_ = np.zeros_like(coef_)
        else:
            best_func_value = np.inf
            np.random.seed(self.random_state)
            for _ in range(self.decimal_max_iter):
                if _==0:
                    noise = np.zeros(len(coef_))
                else:
                    noise = np.random.randn(len(coef_))*np.random.rand()*10
                new_coef = np.round(coef_/self.coef_decimal + noise)*self.coef_decimal
                for bi, bound in enumerate(self.bounds):
                    if bound[0] is not None or bound[1] is not None:
                        new_coef[bi] = np.clip(new_coef[bi], bound[0], bound[1])
                if self.coef_sum is not None and np.abs(np.sum(new_coef*sum_weights)-self.coef_sum)>self.coef_decimal/10.:
                    continue
                func_value = func(new_coef, X2, y2, alpha, self.l1_ratio, sample_weight)[0]#np.r_[new_coef,intercept_]
                if func_value < best_func_value:
                    best_func_value = func_value
                    best_coef = new_coef
            coef_ = best_coef
        
        if self.orders is not None:
            for x in self.orders:
                coef_[x[1]] += coef_[x[0]]
        self.coef_ = coef_.reshape(1,-1)/self.coef_decimal
        self.intercept_ = np.zeros(1)#intercept_.reshape(1,)
        return self
        

class LTRPairwise(BaseEstimator, ClassifierMixin):
    """Learning to rank, pairwise approach
    For each pair A and B, learn a score so that A>B or A<B based on the ordering.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        It must be a classifier with a ``decision_function`` function.
    verbose : bool, optional, defaults to False
        Whether prints more information.
    """
    def __init__(self, estimator, classes,
                    class_weight=None, min_level_diff=1, verbose=False):
        super().__init__()
        self.estimator = estimator
        self.classes = classes
        self.class_weight = class_weight
        self.min_level_diff = min_level_diff
        self.verbose = verbose
        
    #def __setattr__(self, name, value):
    #    setattr(self.estimator, name, value)
    #    super().__setattr__(name, value)
        
    def _generate_pairs(self, X, y, sample_weight):
        X2 = []
        y2 = []
        sw2 = []
        for i, j in combinations(range(len(X)), 2):
            # if there is a tie, ignore it
            if np.abs(y[i]-y[j])<self.min_level_diff:
                continue
            X2.append( X[i]-X[j] )
            y2.append( 1 if y[i]>y[j] else 0 )
            if sample_weight is not None:
                sw2.append( max(sample_weight[i], sample_weight[j]) )
        
        if sample_weight is None:
            sw2 = None
        else:
            sw2 = np.array(sw2)

        return np.array(X2), np.array(y2), sw2

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            if self.class_weight is not None:
                sample_weight = get_sample_weights(y, class_weight=self.class_weight, prior_count=2)
            else:
                sample_weight = np.ones(len(X))
        #sample_weight /= (np.mean(sample_weight)*len(X))
        
        # generate pairs
        X2, y2, sw2 = self._generate_pairs(X, y, sample_weight)
        sw2 = sw2/sw2.mean()
        if self.verbose:
            print('Generated %d pairs from %d samples'%(len(X2), len(X)))

        # fit the model
        self.estimator.fit(X2, y2, sample_weight=sw2)

        # get the mean of z for each level of y
        self.classes_ = self.classes
        z = self.predict_z(X)
        for tol in range(0,len(self.classes_)//2):
            z_means = np.array([z[(y>=cl-tol)&(y<=cl+tol)].mean() for cl in self.classes_])
            if tol==0:
                self.z_means = z_means
            if np.all(np.diff(z_means)>0):
                self.z_means = z_means
                break

        self.coef_ = self.estimator.coef_
        self.intercept_ = np.zeros(1)#self.estimator.intercept_
        return self

    def predict_z(self, X):
        z = self.estimator.decision_function(X)
        return z

    def decision_function(self, X):
        z = self.predict_z(X)
        return z

    def predict_proba(self, X, z=None):
        if z is None:
            z = self.predict_z(X)
        dists = -(z.reshape(-1,1) - self.z_means)**2
        dists[np.isnan(dists)] = -np.inf
        yp = softmax(dists, axis=1)
        return yp

    def predict(self, X):
        yp1d = self.predict_z(X)
        #yp = self.predict_proba(X)
        #yp1d = self.classes_[np.argmax(yp, axis=1)]
        return yp1d


def get_sample_weights(y, class_weight='balanced', prior_count=0):
    assert y.min()==0 ## assume y=[0,1,...]
    K = y.max()+1
    class_weights = {k:1./(np.sum(y==k)+prior_count) for k in range(K)}
    sw = np.array([class_weights[yy] for yy in y])
    sw = sw/np.mean(sw)
    return sw


def get_perf(model_type, y, yp, yp_prob):
    if model_type=='logreg':
        perf = roc_auc_score(y, yp_prob[:,1])
    elif model_type=='ltr':
        perf = spearmanr(y, yp).correlation
    return perf


def get_coef(model_type, model):
    if model_type=='logreg':
        coef = model.base_estimator.coef_.flatten()
    elif model_type=='ltr':
        coef = model.base_estimator.estimator.coef_.flatten()
    return coef
    
    
def stratified_group_k_fold(X, y, groups, K, seed=None):
    """
    https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(K)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    np.random.seed(seed)
    np.random.shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(K):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(K):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices
        
        
def fit_model(X, y, sids, cv_split_, bounds, orders, worst_delirium_mask, model_type='logreg', refit=True, n_jobs=1, best_params=None, random_state=None):
    """
    """
    models = []
    cv_tr_score = []
    cv_te_score = []
    y_yp_te = []
    K = 5
    
    worst_delirium_label = 15
    bounds = [bounds[i] for i in range(len(bounds)) if not worst_delirium_mask[i]]
    
    goodids = np.where(~worst_delirium_mask)[0]
    featureidmap = {goodids[i]:i for i in range(len(goodids))}
    orders = [(featureidmap[x[0]], featureidmap[x[1]]) for x in orders if x[0] in featureidmap and x[1] in featureidmap]
    
    if best_params is None:
        params = []
    else:
        params = best_params
        
    classes = np.arange(y.max()+1)
    n_classes = len(set(classes))
    cv_split = copy.deepcopy(cv_split_)
    Ncv = len(cv_split)
    coef_decimal = 0.5
    coef_sum = 10
        
    # outer CV
    for cvi, cv_sids in enumerate(cv_split):#tqdm
        teids = np.in1d(sids, cv_sids)
        trids = ~teids
        Xtr = X[trids]
        ytr = y[trids]
        if len(set(ytr))!=len(set(y)):
            continue
        
        # standardize
        Xmean = np.nanmean(Xtr, axis=0)
        Xstd = np.nanstd(Xtr, axis=0)
        Xmean[np.isnan(Xmean)] = 0
        Xstd[np.isnan(Xstd)] = 1
        Xstd[Xstd==0] = 1
        # not used since its binary features
        #Xtr = (Xtr-Xmean)/Xstd
        
        # impute missing value
        #imputer = KNNImputer(n_neighbors=K).fit(Xtr)
        #if np.any(np.isnan(Xtr)):
        #    Xtr = imputer.transform(Xtr)
        
        good_ids = np.all(Xtr[:,worst_delirium_mask]!=1, axis=1)
        Xtr = Xtr[good_ids][:,~worst_delirium_mask]
        ytr_full = np.array(ytr)
        ytr = ytr[good_ids]
        
        if model_type=='logreg':
            model_params = {'C':np.logspace(-3,1,5),#[:2],
                            'l1_ratio':np.arange(0.5,1,0.1),#[:2]
                           }
            metric = 'f1_weighted'
            print('###########################################TODO l1')
            model = LogisticRegression(
                        penalty='elasticnet',#TODO l1
                        class_weight='balanced',
                        solver='saga',
                        random_state=random_state,
                        max_iter=1000)
        
        elif model_type=='ltr':
            model_params = {
                #'estimator__C':np.logspace(-3,1,5),#[:2]
                'estimator__l1_ratio':np.arange(0.5,1,0.1),#[:2]
                }
            metric = make_scorer(lambda y,yp:spearmanr(y,yp).correlation)
            model = LTRPairwise(MyLogisticRegression(
                                    class_weight=None,
                                    coef_decimal=coef_decimal,
                                    coef_sum=coef_sum,
                                    random_state=random_state,
                                    max_iter=1000,
                                    bounds=bounds,
                                    orders=orders),
                                classes, class_weight='balanced', min_level_diff=2,
                                verbose=False)
                        
        if best_params is None:
            model.n_jobs = 1
            model = GridSearchCV(model, model_params,
                        n_jobs=n_jobs, refit=True,
                        cv=Ncv, scoring=metric,
                        verbose=False)
        else:
            for p in params[cvi]:
                val = params[cvi][p]
                if '__' in p:
                    pp = p.split('__')
                    exec('model.%s.%s = %f'%(pp[0], pp[1], val))  # TODO assumes two
                else:
                    exec('model.%s = %f'%(p, val))
            model.n_jobs = n_jobs
        model.fit(Xtr, ytr)
        
        if best_params is None and hasattr(model, 'best_params_'):
            params.append({p:model.best_params_[p] for p in model_params})
        if hasattr(model, 'best_estimator_'):
            model = model.best_estimator_
        
        # calibrate
        #model = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
        model = MyCalibrator(model)
        model.fit(Xtr, ytr)
        
        yptr = np.zeros(len(ytr_full))+worst_delirium_label
        yptr[good_ids] = model.predict(Xtr)
        yptr_prob = np.zeros((len(ytr_full), n_classes))
        yptr_prob[:,-1] = 1
        yptr_prob[good_ids] = model.predict_proba(Xtr)
        
        models.append(model)
        cv_tr_score.append(get_perf(model_type, ytr_full, yptr, yptr_prob))
        
        if len(cv_sids)>0:
            Xte = X[teids]
            yte = y[teids]
            sids_te = sids[teids]
            #Xte = (Xte-Xmean)/Xstd  # not used since its binary features
            
            # no missing value
            #try:  # this throws error sometimes, if so, ignore
            #    if np.any(np.isnan(Xte)):
            #        Xte = imputer.transform(Xte)
            #except Exception as ee:
            #    continue
            good_ids = np.all(Xte[:,worst_delirium_mask]!=1, axis=1)
            Xte = Xte[good_ids][:,~worst_delirium_mask]
            yte_full = np.array(yte)
            yte = yte[good_ids]
            
            Xte2 = np.ones((len(yte_full), Xte.shape[1]))
            Xte2[:,model.base_estimator.coef_.flatten()<0] = 0
            ypte_z = model.base_estimator.predict(Xte2)#np.zeros(len(yte_full))+np.inf
            ypte_z[good_ids] = model.base_estimator.predict(Xte)
            #ypte_score = ((ypte_z-model.base_estimator.intercept_.flatten()[0])/coef_decimal).astype(int)
            ypte = np.zeros(len(yte_full)) + worst_delirium_label
            ypte[good_ids] = model.predict(Xte)
            ypte_prob = np.zeros((len(yte_full), n_classes))
            ypte_prob[:,-1] = 1
            ypte_prob[good_ids] = model.predict_proba(Xte)
            
            df_te = pd.DataFrame(
                        data=np.c_[yte_full, ypte_z, ypte, ypte_prob],
                        columns=['y', 'z', 'yp']+['prob(%d)'%k for k in classes])
            df_te['SID'] = sids_te
            y_yp_te.append(df_te)
            cv_te_score.append(get_perf(model_type, yte_full, ypte, ypte_prob))
    
    cv_tr_score = sum(cv_tr_score)/len(cv_tr_score)
    cv_te_score = sum(cv_te_score)/len(cv_te_score)
    
    if refit:
        # standardize
        Xmean = np.nanmean(X, axis=0)
        Xstd = np.nanstd(X, axis=0)
        Xmean[np.isnan(Xmean)] = 0
        Xstd[np.isnan(Xstd)] = 1
        # not used since its binary features
        #X = (X-Xmean)/Xstd
        
        # no missing value
        # impute missing value
        #imputer = KNNImputer(n_neighbors=K).fit(X)
        #if np.any(np.isnan(X)):
        #    X = imputer.transform(X)
        good_ids = np.all(X[:,worst_delirium_mask]!=1, axis=1)
        X = X[good_ids][:,~worst_delirium_mask]
        y_full = np.array(y)
        y = y[good_ids]
        
        if model_type=='logreg':
            print('###########################################TODO l1')
            model = LogisticRegression(
                        penalty='elasticnet', #TODO l1
                        class_weight='balanced',
                        solver='saga',
                        random_state=random_state,
                        max_iter=1000)
        
        elif model_type=='ltr':
            model = LTRPairwise(MyLogisticRegression(
                                    class_weight=None,
                                    coef_decimal=coef_decimal,
                                    coef_sum=coef_sum,
                                    random_state=random_state,
                                    max_iter=1000,
                                    bounds=bounds,
                                    orders=orders),
                                classes, class_weight='balanced', min_level_diff=2,
                                verbose=False)
                        
        for p in params[0]:
            val = Counter([params[cvi][p] for cvi in range(Ncv)]).most_common()[0][0]
            if '__' in p:
                pp = p.split('__')
                exec('model.%s.%s = %f'%(pp[0], pp[1], val))  # TODO assumes two
            else:
                exec('model.%s = %f'%(p, val))
        model.fit(X, y)
            
        # calibrate
        #model = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
        model = MyCalibrator(model)
        model.fit(X, y)
        models.append(model)
        
        X2 = np.ones((len(y_full), X.shape[1]))
        X2[:,model.base_estimator.coef_.flatten()<0] = 0
        yp_z = model.base_estimator.predict(X2)#np.zeros(len(y_full))+np.inf
        yp_z[good_ids] = model.base_estimator.predict(X)
        #yp_score = ((yp_z-model.base_estimator.intercept_.flatten()[0])/coef_decimal).astype(int)
        yp = np.zeros(len(y_full)) + worst_delirium_label
        yp[good_ids] = model.predict(X)
        yp_prob = np.zeros((len(y_full), n_classes))
        yp_prob[:,-1] = 1
        yp_prob[good_ids] = model.predict_proba(X)
        
        df = pd.DataFrame(
                    data=np.c_[y_full, yp_z, yp, yp_prob],
                    columns=['y','z', 'yp']+['prob(%d)'%k for k in classes])
        df['SID'] = sids
        y_yp_te.append(df)
        
    return models, params, cv_tr_score, cv_te_score, y_yp_te
    
    
if __name__=='__main__':
    model_type = 'ltr'
    Ncv = 5
    n_jobs = 2
    Nbt = 0#000
    random_state = 2020
    suffix = '_not_combine_slowing_Aaron'
    
    ## load dataset
    
    with pd.ExcelFile(f'data_to_fit{suffix}.xlsx') as xls:
        dfX = pd.read_excel(xls, 'X')
        dfy = pd.read_excel(xls, 'y')
        df_worst_delirium_names = pd.read_excel(xls, 'worst_delirium_names')
    
    # exclude repeated patients
    """
    exclude_repeated_pts = False
    if exclude_repeated_pts:
        ids = ~np.in1d(dfX.SID, ['AMSD157', 'AMSD159', 'AMSD173'])
        dfX = dfX[ids].reset_index(drop=True)
        dfy = dfy[ids].reset_index(drop=True)
        suffix = '_exclude_repeated_pts'
    else:  
        suffix = ''
    """
    
    sids = dfX.SID.astype(str).values
    mrns = dfX.MRN.astype(str).values
    worst_delirium_Xnames = df_worst_delirium_names.EEGName.values.astype(str)
       
    ## more preprcessing
    X = dfX.drop(columns=['SID','MRN'])
    Xnames = np.array(X.columns)
    X = X.values.astype(float)
    
    yname = 'CAM-S LF (0-19)'
    y = dfy[yname].values.astype(int)
    # for CAM-S LF, combine rare numbers
    if yname == 'CAM-S LF (0-19)':
        y[y>=15]=15
    # before Counter({15: 141, 2: 25, 0: 25, 1: 24, 5: 21, 14: 19, 3: 19, 11: 19, 13: 17, 9: 17, 6: 16, 12: 15, 10: 14, 4: 13, 8: 10, 7: 8, 17: 3, 16: 1})
    # after Counter({15: 145, 2: 25, 0: 25, 1: 24, 5: 21, 14: 19, 3: 19, 11: 19, 13: 17, 9: 17, 6: 16, 12: 15, 10: 14, 4: 13, 8: 10, 7: 8})
    
    """
    # print missing ratio and remove features with many missing
    missing_thres = 0.1
    missing_ratio = [np.mean(pd.isna(dfX[name])) for name in Xnames]
    #missing_ratio_rank = np.argsort(missing_ratio)[::-1]
    #for i in missing_ratio_rank:
    #    print(Xnames[i], missing_ratio[i])
    good_ids = [i for i in range(len(missing_ratio)) if missing_ratio[i]<=missing_thres]
    bad_ids = [i for i in range(len(missing_ratio)) if missing_ratio[i]>missing_thres]
    print('%d out of %d features are removed due to missing ratio > %g%%:\n%s'%(len(bad_ids), len(missing_ratio), missing_thres*100, Xnames[bad_ids]))
    Xnames = Xnames[good_ids]
    X = X[:, good_ids]

    # remove features wiortth small std
    std_thres = 0.01
    stds = np.nanstd(X, axis=0)
    good_ids = stds>std_thres
    bad_ids = ~good_ids
    print('%d out of %d features are removed due to std < %g:\n%s'%(np.sum(bad_ids), X.shape[1], std_thres, Xnames[bad_ids]))
    Xnames = Xnames[good_ids]
    X = X[:, good_ids]
    """
    
    # remove binary features with less than Ncv being 1 or 0
    thres = Ncv
    bin_ids = np.array([i for i in range(X.shape[1]) if set(X[:,i])=={0,1}])
    sum1 = np.nansum(X[:,bin_ids], axis=0)
    bad_ids = bin_ids[sum1<=thres]
    good_ids = [i for i in range(X.shape[1]) if i not in bad_ids]
    print('%d out of %d binary features are removed due to #1 < %g:\n%s'%(len(bad_ids), X.shape[1], thres, Xnames[bad_ids]))
    Xnames = Xnames[good_ids]
    X = X[:, good_ids]
    
    # add worst_delirium_Xnames
    to_add_Xnames = sorted(set(worst_delirium_Xnames) - set(Xnames))
    print('%d features are added:\n%s'%(len(to_add_Xnames), to_add_Xnames))
    Xnames = np.r_[Xnames, to_add_Xnames]
    X = np.c_[X, dfX[to_add_Xnames].values.astype(float)]
    worst_delirium_mask = np.in1d(Xnames, worst_delirium_Xnames)
    
    very_harmful_Xnames = [
        #'GPD (Generalized periodic discharges) (=GPED/PED) (Not triphasic)',
        #'GPD with Triphasic morphology',
        #'Triphasic waves',
        'GPD or BIPD',
        ]
    beneficial_Xnames = [
        #'PDR (Posterior dominant rhythm) (>=8 Hz); If present - specify highest freq.',
        #'Sleep patterns (Spindles, K-complex, Vertex waves)',
        #'Symmetry (e.g. no focal slowing)'
        ]
    notsure_Xnames = [
        #'Excess/Diffuse alpha',
        'Excess/Diffuse beta',
        ]
    # others are harmful
    bounds = []
    for xn in Xnames:
        if xn in very_harmful_Xnames:
            bounds.append((0.1,None))
        elif xn in notsure_Xnames:
            bounds.append((None,None))
        elif xn in beneficial_Xnames:
            bounds.append((None,0))
        else:
            bounds.append((0,None))
    orders = [
        (list(Xnames).index('Generalized/Diffuse theta slowing'), list(Xnames).index('Generalized/Diffuse delta slowing')),
        #(list(Xnames).index('Focal/Unilateral theta slowing'), list(Xnames).index('Focal/Unilateral delta slowing')),
        ]
        
    N, D = X.shape
    print(X.shape, y.shape)
    for xn in Xnames:
        print(f"'{xn}',")
    print(f'{len(Xnames)} features')
    print('label distribution', Counter(y))
    
    # generate CV split
    cv_split_path = 'cv_split_Ncv%d_random_state%d.csv'%(Ncv, random_state)
    if not os.path.exists(cv_split_path):
        cv_split = np.zeros(len(X))
        for cvi, (_, teid) in enumerate(stratified_group_k_fold(X, y, mrns, Ncv, seed=random_state)):
            cv_split[teid] = cvi
        pd.DataFrame(data={'SID':sids, 'MRN':mrns, 'y':y, 'CV':cv_split}).to_csv(cv_split_path, index=False)
    df_cv = pd.read_csv(cv_split_path)
    assert [len(set(df_cv.MRN[df_cv.CV==k])&set(df_cv.MRN[df_cv.CV!=k])) for k in range(Ncv)] == [0]*Ncv
    cv_split = [df_cv.SID[df_cv.CV==i].values for i in range(Ncv)]
    
    # fit model with bootstrap
    tr_scores_bt = []
    te_scores_bt = []
    y_yp_bt = []
    coefs_bt = []
    params = None
    for bti in tqdm(range(Nbt+1)):
        if bti==0:
            ybt = y
            Xbt = X
            sidsbt = sids
        else:
            btids = np.random.choice(len(X), len(X), replace=True)
            ybt = y[btids]
            if len(set(ybt))!=len(set(y)):
                continue
            Xbt = X[btids]
            sidsbt = sids[btids]
        
        models, params, cv_tr_score, cv_te_score, y_yp = fit_model(
                            Xbt, ybt, sidsbt, cv_split,
                            bounds, orders, worst_delirium_mask, refit=True,#bti==0,
                            model_type=model_type, n_jobs=n_jobs,
                            best_params=params,
                            random_state=random_state+bti)
        tr_scores_bt.append(cv_tr_score)
        te_scores_bt.append(cv_te_score)
        y_yp_bt.append(y_yp)
        coefs_bt.append(get_coef(model_type, models[-1]))
        
        if bti==0:
            final_model = models[-1]
            print('tr score', cv_tr_score)
            print('te score', cv_te_score)
            print(pd.DataFrame(data={'name':Xnames[~np.in1d(Xnames, worst_delirium_Xnames)], 'coef':coefs_bt[-1]}).sort_values('coef')[::-1])

    if Nbt>0:
        print('tr score: %f [%f -- %f]'%(tr_scores_bt[0], np.percentile(tr_scores_bt[1:], 2.5), np.percentile(tr_scores_bt[1:], 97.5)))
        print('te score: %f [%f -- %f]'%(te_scores_bt[0], np.percentile(te_scores_bt[1:], 2.5), np.percentile(te_scores_bt[1:], 97.5)))
    else:
        print('tr score: %f'%tr_scores_bt[0])
        print('te score: %f'%te_scores_bt[0])
    
    y_yps = []
    for bti, y_yp in enumerate(y_yp_bt):
        for cvi, y_yp_cv in enumerate(y_yp):
            N = len(y_yp_cv)
            cols = list(y_yp_cv.columns)
            y_yp_cv['bti'] = np.zeros(N)+bti
            if cvi==Ncv:
                y_yp_cv['cvi'] = 'full'
            else:
                y_yp_cv['cvi'] = np.zeros(N)+cvi
            y_yps.append(y_yp_cv[['bti', 'cvi']+cols])
    y_yps = pd.concat(y_yps, axis=0)
    y_yps.to_csv(f'cv_predictions_{model_type}_Nbt{Nbt}{suffix}.csv', index=False)
    
    with open(f'results_{model_type}_Nbt{Nbt}{suffix}.pickle', 'wb') as ff:
        pickle.dump({'tr_scores_bt':tr_scores_bt,
                     'te_scores_bt':te_scores_bt,
                     'coefs_bt':coefs_bt,
                     'y_yp_bt':y_yp_bt,
                     'params':params,
                     'model':final_model,
                     'Xnames':Xnames,
                     'worst_delirium_Xnames':worst_delirium_Xnames
                    }, ff)
                    
