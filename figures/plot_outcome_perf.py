import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import seaborn as sns
sns.set_style('ticks')


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
        
    suffix = '_not_combine_slowing_Aaron'
    with open(f'outcome_model_data_for_boxplot{suffix}.pickle','rb') as ff:
        cv_scores = pickle.load(ff)
    
    keys = [
        ('CAMSLF', 'GOSDisch'),
        ('VECAMS', 'GOSDisch'),
        
        ('CAMSLF', 'DeceasedDisch'),
        ('VECAMS', 'DeceasedDisch'),
        
        ('CAMSLF', 'Deceased3month'),
        ('VECAMS', 'Deceased3month'),
        ]
    metrics = ['AUROC', 'Cohen\'s kappa', 'FPR', 'FNR', 'PPV', 'NPV']
    outcomes = ['GOSDisch', 'DeceasedDisch', 'Deceased3month']
    outcome2txt = {'DeceasedDisch':'In-hospital\nmortality', 'Deceased3month':'3-month\nmortality', 'GOSDisch':'Discharge\nGOS'}
    colors = {'DeceasedDisch':'k', 'Deceased3month':'r', 'GOSDisch':'b'}
    score2txt = {'CAMSLF':'CAM-S LF', 'VECAMS':'VE-CAM-S'}
    metric2ylim = {
        'AUROC':[0.4,1],
        'Cohen\'s kappa':[0,0.8],
        'FPR':[0,0.71],
        'FNR':[0,0.71],
        'PPV':[0.3,1.05],
        'NPV':[0.3,1.05],}
    
    index = list(cv_scores[keys[0]][0].index)
    Nbt = len(cv_scores[keys[0]])
    
    panel_xoffset = -0.12
    panel_yoffset = 1.01
    xticks = [0,1,3,4,6,7]
    xlim = [min(xticks)-0.8, max(xticks)+0.8]
    
    figsize = (12,9)
    # scatter plot
    plt.close()
    fig = plt.figure(figsize=figsize)
    for mi, metric in enumerate(metrics):
        ax = fig.add_subplot(len(metrics)//2, 2, mi+1)
        bp = ax.boxplot(
            [[cv_scores[k][ii].loc[metric].values[0] for ii in range(1,Nbt)] for k in keys],
            whis=[2.5,97.5],
            usermedians=[cv_scores[k][0].loc[metric].values[0] for k in keys],
            positions=xticks,
            labels=[score2txt[k[0]] for k in keys],
            showfliers=False,)
        plt.xticks(rotation=20)
        for item in ['medians']:#'boxes', 'whiskers', 'caps']:#, 'fliers'
            for ki, k in enumerate(keys):
                plt.setp(bp[item][ki], lw=2)#, colors[k[1]])
        #if mi==len(metrics)-1:
        for oi, outcome in enumerate(outcomes):
            yy = 0.3 if metric=='NPV' and outcome in ['DeceasedDisch', 'Deceased3month'] else 1
            ax.text(
                (0.5+3*oi-xlim[0])/(xlim[1]-xlim[0]), yy,
                outcome2txt[outcome], ha='center', va='top',
                transform=ax.transAxes)
            lb1, ub1 = np.percentile([cv_scores[keys[oi*2]][ii].loc[metric].values[0] for ii in range(1,Nbt)], (2.5,97.5))
            lb2, ub2 = np.percentile([cv_scores[keys[oi*2+1]][ii].loc[metric].values[0] for ii in range(1,Nbt)], (2.5,97.5))
            max_ub = max(ub1,ub2)
            sig_shift = 0.015
            sig_height = 0.01
            if lb1>ub2 or lb2>ub1:
                ax.plot(
                    [3*oi,3*oi,3*oi+1,3*oi+1],
                    [max_ub+sig_shift, max_ub+sig_shift+sig_height, max_ub+sig_shift+sig_height, max_ub+sig_shift],
                    c='k', lw=2)
                ax.text(3*oi+0.5, max_ub+sig_shift+sig_height, '*', ha='center', va='center', fontweight='bold', fontsize=20)
        ax.axvline(2, ls='--', c='k', lw=1)
        ax.axvline(5, ls='--', c='k', lw=1)
        ax.yaxis.grid(True)
        ax.set_ylabel(metric)
        ax.set_ylim(metric2ylim[metric])
        ax.set_xlim(xlim)
        ax.text(panel_xoffset, panel_yoffset, chr(ord('A')+mi), ha='right', va='top', transform=ax.transAxes, fontweight='bold')
        sns.despine()
    
    plt.tight_layout()
    if display_type=='pdf':
        plt.savefig('outcome_performance.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig('outcome_performance.png', bbox_inches='tight', pad_inches=0.05)
    elif display_type=='tiff':
        plt.savefig('outcome_performance.tiff', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
