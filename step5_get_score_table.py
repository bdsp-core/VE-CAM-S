from itertools import combinations
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from mord import LogisticAT
import matplotlib.pyplot as plt
from step2_fit_model_delirium import MyCalibrator, LTRPairwise, MyLogisticRegression


if __name__=='__main__':
    model_type = 'ltr'
    with open(f'results_{model_type}_Nbt0.pickle', 'rb') as ff:
        res = pickle.load(ff)
    for k in res:
        exec('%s = res[\'%s\']'%(k,k))
    random_state = 2020
    
    df = pd.read_csv('step4_output_df.csv')
    outcomes = ['Deceased3month', 'DeceasedDisch', 'GOSDisch']
    family = ['binomial', 'binomial', 'ordinal']
    models = {}
    for i, outcome in enumerate(outcomes):
        if family[i]=='binomial':
            model_ = LogisticRegression(penalty='none', class_weight='balanced', random_state=random_state, max_iter=1000)
            model_.fit(df.VECAMS.values.reshape(-1,1), df[outcome].values)
            model_ = CalibratedClassifierCV(base_estimator=model_, method='sigmoid', cv='prefit')
            model_.fit(df.VECAMS.values.reshape(-1,1), df[outcome].values)
        elif family[i]=='ordinal':
            model_ = LogisticAT(alpha=0)
            model_.fit(df.VECAMS.values.reshape(-1,1), df[outcome].values)
            model_ = MyCalibrator(model_)
            model_.fit(df.VECAMS.values.reshape(-1,1), df[outcome].values)
        models[outcome] = model_
    
    #intercept = model.base_estimator.estimator.intercept_[0]
    scores = model.base_estimator.estimator.coef_[0].astype(int)
    unique_scores = set()
    scores = scores[scores>0]
    for k in range(0, len(scores)+1):
        for score_comb in combinations(scores, k):
            unique_scores.add(sum(score_comb))
    unique_scores = sorted(unique_scores)
    print(f'unique_scores = {unique_scores}')
    
    #yps = []
    #ypps = []
    yp_ints = []
    mortality_dischs = []
    mortality_3months = []
    gos_dischs = []
    for score in unique_scores:
        z = np.array([[score]])#+intercept
        #yps.append( model.predict(None, z=z)[0] )
        ypp = model.base_estimator.predict_proba(None, z=z)[0]
        yp_ints.append( np.argsort(ypp)[[-1,-2]] )
        mortality_dischs.append( models['DeceasedDisch'].predict_proba(z)[0,1]*100 )
        mortality_3months.append( models['Deceased3month'].predict_proba(z)[0,1]*100 )
        gos_dischs.append( models['GOSDisch'].predict_proba(z)[0]*100 )
    gos_dischs = np.array(gos_dischs)
    yp_ints = np.array(yp_ints)
    
    df_lut = pd.DataFrame(
            data=np.c_[unique_scores, yp_ints, mortality_dischs, mortality_3months, gos_dischs],
            columns=['VE-CAM-S', 'CAM-S LF(1)', 'CAM-S LF(2)', 'DeceasedDisch', 'Deceased3month']+[f'GOSDisch({x})' for x in range(1,6)])
    df_lut['GOSDisch(<=3)'] = df_lut['GOSDisch(1)'] + df_lut['GOSDisch(2)'] + df_lut['GOSDisch(3)']
    df_lut.to_excel('step5_output_lookuptable.xlsx', index=False)
    
    #plt.plot(df_lut['VE-CAM-S'], df_lut['Deceased3month']/100);plt.plot(df_lut['VE-CAM-S'],[np.mean(df['Deceased3month'][(df.VECAMS>=x-1)&(df.VECAMS<=x+1)]) for x in range(0,21)]);plt.show()
    #plt.plot(df_lut['VE-CAM-S'], df_lut['GOSDisch(5)']/100);plt.plot(df_lut['VE-CAM-S'],[np.mean(df['GOSDisch'][(df.VECAMS>=x-1)&(df.VECAMS<=x+1)]==5) for x in range(0,21)]);plt.show()
