from collections import defaultdict
import pickle
import numpy as np
from scipy.stats import linregress
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score
from sklearn.calibration import calibration_curve
from step2_fit_model_delirium import MyCalibrator, LTRPairwise, MyLogisticRegression


def caibration_slope(y, yp):
    obs, pred = calibration_curve(y, yp, n_bins=10, strategy='quantile')
    cslope, intercept, _, _, _ = linregress(pred, obs)
    return cslope
    
    
if __name__=='__main__':
    model_type = 'ltr'
    Ncv = 5
    random_state = 2020
    suffix = '_not_combine_slowing_Aaron'
    
    with open(f'results_{model_type}_Nbt0{suffix}.pickle', 'rb') as ff:
        res = pickle.load(ff)
    for k in res:
        exec('%s = res[\'%s\']'%(k,k))

    df_pred_all = pd.read_csv(f'cv_predictions_{model_type}_Nbt0{suffix}.csv')
    df_pred_all['SID'] = df_pred_all.SID.astype(str)
    df_pred_all = df_pred_all[(df_pred_all.cvi!='full')&(df_pred_all.bti==0)].reset_index(drop=True)

    df_all = pd.read_excel('mastersheet_combined_reformatted.xlsx')
    # rename columns
    df_all = df_all.rename(columns={k:k.strip() for k in df_all.columns})
    df_all = df_all.rename(columns={'Total Points (Add age score to comorbidy score)' : 'CCI'})
    # exclude two patients without EEG
    df_all = df_all[(df_all.SID!='AMSD086') & (df_all.SID!='AMSD153')].reset_index(drop=True)
    df_all['MRN'] = df_all.MRN.astype(str)
    two_mrn_ids = np.where(df_all.MRN.str.contains('\('))[0]
    df_all.loc[two_mrn_ids, 'MRN'] = df_all.MRN.iloc[two_mrn_ids].str.replace('(','_').str.replace(')','').str.split('_', expand=True)[1]
    underscore_ids = np.where(df_all.MRN.str.contains('_'))[0]
    df_all.loc[underscore_ids, 'MRN'] = df_all.MRN.iloc[underscore_ids].str.replace('_','')
    df_all['SID'] = df_all.SID.astype(str)
    df_all['Type of EEG (routine, LTM)'] = df_all['Type of EEG (routine, LTM)'].astype(str).str.strip()
    df_all['Race'] = df_all.Race.astype(str).str.strip()
    assert np.all(df_all.MRN.str.isnumeric())
    assert len(set(df_all.SID))==len(df_all)
        
    ids = [np.where(df_all.SID==x)[0][0] for x in df_pred_all.SID]
    df_all = df_all.iloc[ids].reset_index(drop=True)

    time_from_assess_col = 'If not in epoch/during EEG -> Give time (min) from closest epoch/ active EEG'
    df_all.loc[df_all[time_from_assess_col].astype(str).str.strip()=='NA', time_from_assess_col] = 0
    df_all.loc[pd.isna(df_all[time_from_assess_col]), time_from_assess_col] = 0
    df_all[time_from_assess_col] = df_all[time_from_assess_col].astype(float)

    df = pd.concat([df_all, df_pred_all.drop(columns='SID')], axis=1)
    
    ## subset analysis

    #cohen_kappas = []
    spearmanrs = defaultdict(list)
    aucs = defaultdict(list)
    css = defaultdict(list)
    Ns = defaultdict(list)
    K = 16
    thres = 5
    np.random.seed(random_state)
    Nbt = 1000
    for bti in tqdm(range(Nbt+1)):
        if bti==0:
            df_bt = df.copy()
        else:
            btids = np.random.choice(len(df), len(df), replace=True)
            df_bt = df.iloc[btids].reset_index(drop=True)
            
        try:
            y = df_bt.y.values
            yp = df_bt.z.values
            ypp = df_bt[[f'prob({x})' for x in range(K)]].values
            Ns['all'].append(len(y))
            spearmanrs['all'].append(spearmanr(y, yp)[0])
            aucs['all'].append(roc_auc_score((y>=thres).astype(int), ypp[:,thres:].sum(axis=1)))
            css['all'].append(caibration_slope((y>=thres).astype(int), ypp[:,thres:].sum(axis=1)))
            #cohen_kappas.append(cohen_kappa_score(y, yp_int))

            # Ryan vs Eyal
            #ids = df_bt.SID.astype(str).str.startswith('AMSD')
            #Ns['ryan'].append(len(y[~ids]))
            #Ns['eyal'].append(len(y[ids]))
            #spearmanrs['ryan'].append(spearmanr(y[~ids], yp[~ids])[0])
            #spearmanrs['eyal'].append(spearmanr(y[ids], yp[ids])[0])

            # young vs middle vs old
            for subset in ['young', 'middle', 'old']:
                if subset=='young':
                    ids = df_bt.Age<40
                elif subset=='middle':
                    ids = (df_bt.Age>=40)&(df_bt.Age<60)
                else:
                    ids = df_bt.Age>=60
                Ns[subset].append(len(y[ids]))
                spearmanrs[subset].append(spearmanr(y[ids], yp[ids])[0])
                aucs[subset].append(roc_auc_score((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))
                css[subset].append(caibration_slope((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))

            # male vs female
            for subset in ['male', 'female']:
                if subset=='male':
                    ids = df_bt.Gender.astype(str).str.strip().str.upper()=='M'
                else:
                    ids = df_bt.Gender.astype(str).str.strip().str.upper()=='F'
                Ns[subset].append(len(y[ids]))
                spearmanrs[subset].append(spearmanr(y[ids], yp[ids])[0])
                aucs[subset].append(roc_auc_score((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))
                css[subset].append(caibration_slope((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))

            # race 
            for subset in ['White', 'Black or African American']:
                ids = df_bt.Race==subset
                Ns[subset].append(len(y[ids]))
                spearmanrs[subset].append(spearmanr(y[ids], yp[ids])[0])
                aucs[subset].append(roc_auc_score((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))
                css[subset].append(caibration_slope((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))
            
            # ICU vs non-ICU
            for subset in ['ICU', 'NonICU']:
                if subset=='ICU':
                    ids = df_bt['Unit Type'].astype(str).str.lower().str.contains('icu')
                else:
                    ids = ~ids
                spearmanrs[subset].append(spearmanr(y[ids], yp[ids])[0])
                Ns[subset].append(len(y[ids]))
                aucs[subset].append(roc_auc_score((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))
                css[subset].append(caibration_slope((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))
            
            # routine vs LTM
            for subset in ['LTM', 'Routine']:
                if subset=='LTM':
                    ids = df_bt['Type of EEG (routine, LTM)']=='LTM'
                else:
                    ids = df_bt['Type of EEG (routine, LTM)']=='Routine'
                Ns[subset].append(len(y[ids]))
                spearmanrs[subset].append(spearmanr(y[ids], yp[ids])[0])
                aucs[subset].append(roc_auc_score((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))
                css[subset].append(caibration_slope((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))
            
            """
            # CAM-ICU 0 vs 1
            for subset in ['CAM-ICU = 0', 'CAM-ICU = 1']:
                if subset=='CAM-ICU = 0':
                    ids = df_bt['CAM-ICU (0/1)']==0
                else:
                    ids = df_bt['CAM-ICU (0/1)']==1
                Ns[subset].append(len(y[ids]))
                spearmanrs[subset].append(spearmanr(y[ids], yp[ids])[0])
                aucs[subset].append(roc_auc_score((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))
                css[subset].append(caibration_slope((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))
            """
            
            # not comatose
            subset = 'Not comatose'
            ids = df_bt['Assign full penalty of 7 d/t coma (0=N; 1=Y)']==0
            ids |= (df_bt.SID=='130') # SID 130 is not comatose
            Ns[subset].append(len(y[ids]))
            spearmanrs[subset].append(spearmanr(y[ids], yp[ids])[0])
            aucs[subset].append(roc_auc_score((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))
            css[subset].append(caibration_slope((y[ids]>=thres).astype(int), ypp[ids][:,thres:].sum(axis=1)))
            
            """
            #ids = df_bt['Assign full penalty of 7 d/t coma (0=N; 1=Y)']==1
            #spearmanrs['Coma'].append(spearmanr(y[ids], yp[ids])[0])
            #spearmanrs['Non-Coma'].append(spearmanr(y[~ids], yp[~ids])[0])
            #Ns['Coma'].append(len(y[ids]))
            #Ns['Non-Coma'].append(len(y[~ids]))
            
            # assessment time from EEG
            ids = df_bt[time_from_assess_col]==0
            spearmanrs['Inside epoch'].append(spearmanr(y[ids], yp[ids])[0])
            Ns['Inside epoch'].append(len(y[ids]))
            ids = (df_bt[time_from_assess_col]>0) & (df_bt[time_from_assess_col]<=30)
            spearmanrs['0<assessment time from EEG<=30min'].append(spearmanr(y[ids], yp[ids])[0])
            Ns['0<assessment time from EEG<=30min'].append(len(y[ids]))
            ids = (df_bt[time_from_assess_col]>30) & (df_bt[time_from_assess_col]<=60)
            spearmanrs['30min<assessment time from EEG<=1h'].append(spearmanr(y[ids], yp[ids])[0])
            Ns['30min<assessment time from EEG<=1h'].append(len(y[ids]))
            ids = df_bt[time_from_assess_col]>60
            spearmanrs['assessment time from EEG>1h'].append(spearmanr(y[ids], yp[ids])[0])
            Ns['assessment time from EEG>1h'].append(len(y[ids]))
            """
        except Exception as ee:
            print(str(ee))
            continue

    values = defaultdict(list)
    metrics = ['spearmanr', 'auc', 'cs']
    for key in spearmanrs:
        values['subset'].append(key)
        values['N'].append(Ns[key][0])
        for metric_ in metrics:
            metric = eval(metric_+'s')
            val = metric[key][0]
            if Nbt>1:
                lb = np.percentile(metric[key][1:], 2.5)
                ub = np.percentile(metric[key][1:], 97.5)
            else:
                lb = np.nan
                ub = np.nan
            values[metric_].append(f'{val:.2f} ({lb:.2f} -- {ub:.2f})')
    df_perf_subset = pd.DataFrame(values)[['subset','N']+metrics]
    df_perf_subset.to_csv(f'perf_subset{suffix}.csv', index=False)

    ## save coef

    coef = model.base_estimator.estimator.coef_.flatten()
    df_coef = pd.DataFrame(data={'feature':Xnames[~np.in1d(Xnames, worst_delirium_Xnames)], 'coef':coef})
    df_coef = df_coef.sort_values('coef', ascending=False).reset_index(drop=True)
    df_coef.to_csv(f'coef_cam-s-lf{suffix}.csv', index=False)
