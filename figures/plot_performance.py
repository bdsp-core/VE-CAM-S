import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress, spearmanr
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
sns.set_style('ticks')


def bootstrap_curves(x, xs, ys, bounds=None, verbose=True):
    _, idx = np.unique(x, return_index=True)
    idx = np.sort(idx)
    x = x[idx]
    idx = np.argsort(x)
    x_res = x[idx]
    
    ys_res = []
    for _ in tqdm(range(len(xs)), disable=not verbose):
        try:
            xx = xs[_]
            yy = ys[_]
            _, idx = np.unique(xx, return_index=True)
            idx = np.sort(idx)
            xx = xx[idx]; yy = yy[idx]
            idx = np.argsort(xx)
            xx = xx[idx]; yy = yy[idx]
            foo = interp1d(xx, yy, kind='linear')
            ys_res.append( foo(x) )
        except Exception as ee:
            print(str(ee))
            continue
    ys_res = np.array(ys_res)
    if bounds is not None:
        ys_res = np.clip(ys_res, bounds[0], bounds[1])
        
    return x_res, ys_res
    

if __name__ == '__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        elif 'tiff' in sys.argv[1].lower():
            display_type = 'tiff'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/tiff/pdf'%__file__)
        
    K = 16
    suffix = '_not_combine_slowing_Aaron'
    model_type = 'ltr'
    df_pred = pd.read_csv(f'../cv_predictions_{model_type}_Nbt0{suffix}.csv')
    df_pred['SID'] = df_pred.SID.astype(str)
    df_pred = df_pred[(df_pred.cvi!='full')&(df_pred.bti==0)].reset_index(drop=True)
    y = df_pred.y.values
    yp = df_pred.z.values
    
    # get CAM = 0 / CAM = 1 / comatose info
    df2 = pd.read_excel('../mastersheet_combined_reformatted.xlsx')
    df2['SID'] = df2.SID.astype(str)
    df2 = df2.rename(columns={'CAM algorithm (CAM 1 & 2 , and 3 or 4 SATISFIED - DELIRIUM POSITIVE)':'CAM', 'Assign full penalty of 7 d/t coma (0=N; 1=Y)':'Coma'})
    df2.loc[df2.SID=='130', 'Coma'] = 0
    df_pred = df_pred.merge(df2[['SID', 'CAM', 'Coma']], on='SID', how='left')
    
    Nbt = 1000
    random_state = 2020
    np.random.seed(random_state)
    ys = []; yps = []; yp_probs = []
    for bti in tqdm(range(Nbt+1)):
        if bti==0:
            df_bt = df_pred.copy()
        else:
            btids = np.random.choice(len(df_pred), len(df_pred), replace=True)
            df_bt = df_pred.iloc[btids].reset_index(drop=True)
        ys.append( df_bt.y.values )
        yps.append( df_bt.z.values )
        yp_probs.append( df_bt[[f'prob({x})' for x in range(K)]].values )
        #yps_int.append( np.argmax(yp_probs, axis=1) )
    corrs = [spearmanr(ys[i], yps[i])[0] for i in range(len(ys))]
    corr = corrs[0]
    corr_lb, corr_ub = np.percentile(corrs, (2.5, 97.5))
    
    panel_xoffset = -0.12
    panel_yoffset = 1.01
    
    figsize = (11,9.5)
    # scatter plot
    plt.close()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, :])
    bp = ax.boxplot([yp[y==i] for i in range(K)], positions=np.arange(K), showfliers=False)
    plt.setp(bp['medians'], color='r', lw=2)
    
    coma_ids = df_pred.Coma==1
    gray = (20/255, 20/255, 20/255)
    ax.scatter(y[coma_ids]+np.random.randn(len(y[coma_ids]))/20., yp[coma_ids]+np.random.randn(len(yp[coma_ids]))/10., s=50, ec='none', alpha=0.4, fc=gray)
    
    cam0_ids = (df_pred.Coma==0)&(df_pred.CAM==0)
    skyblue = (86/255, 180/255, 233/255)
    ax.scatter(y[cam0_ids]+np.random.randn(len(y[cam0_ids]))/20., yp[cam0_ids]+np.random.randn(len(yp[cam0_ids]))/10., s=50, ec='none', alpha=0.4, fc=skyblue)
    
    cam1_ids = (df_pred.Coma==0)&(df_pred.CAM==1)
    reddish_purple = (204/255, 121/255, 167/255)
    ax.scatter(y[cam1_ids]+np.random.randn(len(y[cam1_ids]))/20., yp[cam1_ids]+np.random.randn(len(yp[cam1_ids]))/10., s=50, ec='none', alpha=0.4, fc=reddish_purple)
    
    ax.text(0.03, 0.91, f'Spearman\'s correlation R = {corr:.2f} ({corr_lb:.2f} -- {corr_ub:.2f})',
            ha='left', va='top', transform=ax.transAxes)
            
    ax.scatter([100], [100], ec='none', fc=gray, label='Coma')
    ax.scatter([100], [100], ec='none', fc=skyblue, label='No delirium or coma')
    ax.scatter([100], [100], s=50, ec='none', fc=reddish_purple, label='Delirium')
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=(0, 0.6))
    ax.set_xlabel('CAM-S LF score')
    ax.set_ylabel('VE-CAM-S score')
    ax.set_xlim([-1,16])
    ax.set_ylim([-1,21])
    ax.yaxis.grid(True)
    sns.despine()
    ax.text(panel_xoffset/2, panel_yoffset, 'A', ha='right', va='top', transform=ax.transAxes, fontweight='bold')

    """
    # confusion matrix plot
    cf = confusion_matrix(ys, yps_int)
    plt.close()
    fig=plt.figure(figsize=(12,9))
    ax=fig.add_subplot(111)
    sns.heatmap(np.flipud(cf.T),annot=True,cmap='Blues',fmt='d')
    ax.set_yticklabels(np.arange(K)[::-1])
    ax.set_ylabel('Predicted CAM-S')
    ax.set_xlabel('Actual CAM-S')
    plt.tight_layout()
    if display_type=='pdf':
        plt.savefig(f'confusionmatrix_{model_type}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig(f'confusionmatrix_{model_type}.png', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
    """

    # AUC
    
    levels = list(np.arange(1,K))
    aucs = []
    aucs_lb = []
    aucs_ub = []
    Ns = []
    level_todraw = 4
    for level in tqdm(levels):
        aucs_bt = []
        fprs = []
        tprs = []
        for bti in range(Nbt+1):
            ids = (ys[bti]==0)|(ys[bti]>=level)
            y_ = (ys[bti][ids]>=level).astype(int)
            yp_ = yp_probs[bti][ids][:,level:].sum(axis=1)
            aucs_bt.append(roc_auc_score(y_,yp_))
            
            fpr_, tpr_, tt = roc_curve(y_, yp_)
            fprs.append(fpr_)
            tprs.append(tpr_)
            if bti==0:
                Ns.append(((y_==0).sum(), (y_==1).sum()))
        if Nbt>0:
            auc_lb, auc_ub = np.percentile(aucs_bt[1:], (2.5, 97.5))
        else:
            auc_lb = np.nan
            auc_ub = np.nan
        aucs.append(aucs_bt[0])
        aucs_lb.append(auc_lb)
        aucs_ub.append(auc_ub)
        
        if level==level_todraw:
            fpr, tprs_bt = bootstrap_curves(fprs[0], fprs, tprs, bounds=[0,1], verbose=False)
            tpr_lb, tpr_ub = np.percentile(tprs_bt[1:], (2.5, 97.5), axis=0)
    ax = fig.add_subplot(gs[1,0])
    ax.fill_between(levels, aucs_lb, aucs_ub, color='k', alpha=0.2, label='95% CI')
    ax.plot(levels, aucs, c='k', lw=2, marker='o')
    ax.annotate('',
        (level_todraw+3.05, aucs[levels.index(level_todraw)]-0.03), 
        xytext=(level_todraw+0.35, aucs[levels.index(level_todraw)]-0.01),
        arrowprops=dict(color='k', width=2, headwidth=6))
    
    ax.legend(loc='upper left', frameon=False)
    ax.set_xticks(levels)
    ax.set_xlim(levels[0]-0.1, levels[-1]+0.1)
    ax.set_ylim(0.61, 1.0)
    ax.set_yticks([0.7,0.8,0.9,1])
    ax.set_ylabel('AUROC')
    ax.set_xlabel('Comparison level x (CAM-S=0 vs. CAM-S$\geq$x)')
    ax.yaxis.grid(True)
    sns.despine()
    
    axins = ax.inset_axes([0.53, 0.08, 0.47, 0.47])
    
    axins.fill_between(fpr, tpr_lb, tpr_ub, color='k', alpha=0.2, label='95% CI')
    axins.plot(fpr, tprs_bt[0], c='k', lw=2)#, label=f'CAM-S LF <= {level} vs. >={level+1}:\nAUC = {auc:.2f} [{auc_lb:.2f} - {auc_ub:.2f}]')# (n={np.sum(y2[0]==0)})
        
    axins.plot([0,1],[0,1],c='k',ls='--')
    #axins.legend(loc='lower right', frameon=False)
    axins.set_xlim([-0.01, 1.01])
    axins.set_ylim([-0.01, 1.01])
    axins.set_xticks([0,0.25,0.5,0.75,1])
    axins.set_yticks([0,0.25,0.5,0.75,1])
    axins.set_xticklabels(['0','0.25','0.5','0.75','1'])
    axins.set_yticklabels(['0','0.25','0.5','0.75','1'])
    axins.set_ylabel('Sensitivity')
    axins.set_xlabel('1 - Specificity')
    axins.xaxis.set_label_coords(0.5, 0.15)
    axins.grid(True)
    sns.despine()

    ax.text(panel_xoffset, panel_yoffset, 'B', ha='right', va='top', transform=ax.transAxes, fontweight='bold')

    # calibration
    ax = fig.add_subplot(gs[1,1])
    ax.plot([0,1],[0,1],c='k',ls='--')
    #levels = np.arange(K-1)
    levels = [4]
    for i in levels:
        y2 = [(ys[bti]>i).astype(int) for bti in range(Nbt+1)]
        yp2 = [yp_probs[bti][:, i+1:].sum(axis=1) for bti in range(Nbt+1)]
        obss_preds = [calibration_curve(y2[bti], yp2[bti], n_bins=10, strategy='quantile') for bti in range(Nbt+1)]
        obss = [x[0] for x in obss_preds]
        preds = [x[1] for x in obss_preds]
        #pred, obss = bootstrap_curves(preds[0], preds, obss)#, bounds=[0,1])
        obs = obss[0]
        pred = preds[0]
        cslopes = [linregress(x[1], x[0])[0] for x in obss_preds]
        cslope, intercept, _, _, _ = linregress(pred, obs)
        if Nbt>0:
            cslope_lb, cslope_ub = np.percentile(cslopes[1:], (2.5, 97.5))
            obs_lb, obs_ub = np.percentile(obss[1:], (2.5, 97.5), axis=0)
        else:
            cslope_lb = np.nan
            cslope_ub = np.nan
        #if Nbt>0:
        #    ax.fill_between(pred, obs_lb, obs_ub, color='k', alpha=0.2, label='95% CI')
        ax.plot(pred, cslope*pred+intercept, c='k', lw=2, label=f'CAM-S LF $\leq$ {i} vs. $\geq${i+1}:\ncalib. slope = {cslope:.2f} [{cslope_lb:.2f} - {cslope_ub:.2f}]')# (n={np.sum(y2[0]==0)})
        ax.scatter(pred, obs, c='k', marker='o', s=40)
        
    ax.legend(loc='upper left', frameon=False)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed fraction')
    ax.grid(True)
    ax.text(panel_xoffset, panel_yoffset, 'C', ha='right', va='top', transform=ax.transAxes, fontweight='bold')
    sns.despine()
    
    plt.tight_layout()
    if display_type=='pdf':
        plt.savefig(f'performance_{model_type}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig(f'performance_{model_type}.png', bbox_inches='tight', pad_inches=0.05)
    elif display_type=='tiff':
        plt.savefig(f'performance_{model_type}.tiff', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
    
