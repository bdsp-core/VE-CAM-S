import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
import scipy.cluster.hierarchy as sch
import pandas as pd
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
seaborn.set_style('ticks')


def cluster_corr(corr_array):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    return corr_array[idx, :][:, idx], idx


def binary_correlation(x, y):
    """
    Correlation for binary vectors (0 or 1)
    Ref:
    [1] Zhang, B. and Srihari, S.N., 2003, September. Properties of binary vector dissimilarity measures. In Proc. JCIS Int'l Conf. Computer Vision, Pattern Recognition, and Image Processing (Vol. 1). https://cedar.buffalo.edu/papers/articles/CVPRIP03_propbina.pdf
    [2] Tubbs, J.D., 1989. A note on binary template matching. Pattern Recognition, 22(4), pp.359-365.
    """
    #corr = np.mean(X[:,i]==X[:,j])*2-1
    #corrpearsonr(X[:,i], X[:,j])[0]
    
    s11 = np.sum(x*y)
    s10 = np.sum(x*(1-y))
    s01 = np.sum((1-x)*y)
    s00 = np.sum((1-x)*(1-y))
    
    # The "Correlation" method in Table 1 of [1].
    sigma = np.sqrt( (s10+s11)*(s01+s00)*(s11+s01)*(s00+s10) )
    corr = (s11*s00-s10*s01)/sigma
    
    """
    # The "Rogers-Tanmot" method in Table 1 of [1].
    corr = (s11+s00)/(s11+s00+2*s10+2*s01)
    corr = corr*2-1  # turn it into -1 to 1
    """
    
    return corr
    
    
if __name__=='__main__':
    if len(sys.argv)>=2:
        if 'pdf' in sys.argv[1].lower():
            display_type = 'pdf'
        elif 'png' in sys.argv[1].lower():
            display_type = 'png'
        elif 'tiff' in sys.argv[1].lower():
            display_type = 'png'
        else:
            display_type = 'show'
    else:
        raise SystemExit('python %s show/png/tiff/pdf'%__file__)
    
    df = pd.read_excel('../data_to_fit_not_combine_slowing_Aaron.xlsx', sheet_name='X')
    Xnames = list(df.columns)
    Xnames.remove('SID')
    Xnames.remove('MRN')
    Xnames = np.array(Xnames)

    name2shortname = {
        'PDR (Posterior dominant rhythm) (>=8 Hz); If present - specify highest freq.':'PDR',
        'Sleep patterns (Spindles, K-complex, Vertex waves)':'Sleep',
        'Symmetry (e.g. no focal slowing)':'Symmetry',
        'no PDR (Posterior dominant rhythm) (>=8 Hz); If present - specify highest freq.':'no PDR',
        'no Sleep patterns (Spindles, K-complex, Vertex waves)':'No sleep pattern',
        'no Symmetry (e.g. no focal slowing)':'Asymmetry',
        
        'Generalized/Diffuse delta slowing':'G delta slowing',
        'Generalized/Diffuse theta slowing':'G theta slowing',
        'Generalized/Diffuse delta or theta slowing or GRDA':'GRDA/G delta/theta slow',
        'Diffuse slowing - Either theta or delta or GRDA':'Diffuse slowing',
        'Excess/Diffuse alpha':'Excess alpha',
        'Excess/Diffuse beta':'Excess beta',
        'Focal/Unilateral delta slowing':'F delta slowing',
        'Focal/Unilateral theta slowing':'F theta slowing',
        'Focal slowing - Either theta or delta or LRDA':'Focal slowing',
        'GRDA (Generalized rhythmic delta activity) (= FIRDA - frontal intermittent rhythmic delta activity)':'GRDA',
        'LRDA (Lateralized rhythmic delta activity)':'LRDA',
        'Extreme delta brush':'EDB',

        'Periodic discharges - LPD or GPD or BiPD':'LPD/GPD/BiPD',
        'Any IIIC: LPD or GPD or BiPD or LRDA or Sz or NCSE or TPW (TPW or GPD with TP morphology)':'Any IIIC',
        'GPD or BIPD':'GPD/BIPD',
        'LPD (Lateralized periodic discharges) (=PLED - Periodic lateralized epileptiform discharges)':'LPD',
        'GPD (Generalized periodic discharges) (=GPED/PED) (Not triphasic)':'GPD w/o TPW',
        'GPD with Triphasic morphology':'GPD w TPW',
        'Triphasic waves':'TPW',
        'GPD':'GPD',
        'Sporadic epileptiform discharges (=sporadic discharges)':'Sporadic discharges',
        'BIPD (bilateral indep. periodic discharges) (=BIPLED - Bilateral independent periodic lateralized epileptiform discharges)':'BIPD',
        'BIRDs (brief potentially ictal rhythmic discharges)':'BIRDs',

        'Discrete seizures: generalized':'G Sz', 
        'Discrete seizures: focal':'F Sz',
        'Non convulsive status epilepticus: generalized':'G NCSE',
        'Non convulsive status epilepticus: focal':'F NCSE',

        'Burst suppression with epileptiform activity':'BS w spike',
        'Burst suppression without epileptiform activity':'BS w/o spike',
        'Intermittent brief attenuation':'IBA',
        'Moderately low voltage':'MLV',
        'Extremely low voltage / electrocerebral silence':'ELV',
        'EEG Unreactive':'Unreactive'}
    
    X = df[Xnames].values.astype(float)
    corr = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            corr[i,j] = binary_correlation(X[:,i], X[:,j])
            #corr[i,j] = pearsonr(X[:,i], X[:,j])[0]
    corr, idx = cluster_corr(corr)
    Xnames_to_show = np.array([name2shortname.get(x,x) for x in Xnames])
    
    figsize = (12.5, 6)
    panel_xoffset = -0.1
    panel_yoffset = 1
    
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    corr_linkage = sch.linkage(corr+1, 'average')
    dendro = sch.dendrogram( corr_linkage, ax=ax1, labels=Xnames_to_show[idx])#, truncate_mode='level', p=10)
    dendro_idx = np.arange(0, len(dendro['ivl']))
    ax1.set_ylabel('Distance')
    ax1.set_yticks([])
    ax1.set_xticks(dendro_idx*10)
    ax1.set_xticklabels(dendro['ivl'], rotation=-55, ha='left', fontsize=16)
    seaborn.despine()
    ax1.text(-0.04, panel_yoffset, 'A', ha='right', va='top', transform=ax1.transAxes, fontweight='bold')
    
    im = ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']], cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation=-55, ha='left')
    ax2.set_yticklabels(dendro['ivl'])
    """
    ids = [list(Xnames_to_show).index(x) for x in [
        'GPD/BIPD', 'G delta slowing', 'G theta slowing',  'GRDA',  'G NCSE', 'G Sz',
        'Asymmetry', 'LPD', 'LRDA', 'F NCSE', 'BIRDs',
        'IBA', 'MLV', 'ELV','BS w spike', 'BS w/o spike',
        'EDB', 'Unreactive',
        'No sleep pattern',]]
    im = ax2.imshow(corr[ids][:,ids], cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xticks(np.arange(corr.shape[1]))
    ax2.set_yticks(np.arange(corr.shape[0]))
    ax2.set_xticklabels(Xnames_to_show[ids], rotation=-55, ha='left')
    ax2.set_yticklabels(Xnames_to_show[ids])
    """
    ax2.text(-0.3, panel_yoffset, 'B', ha='right', va='top', transform=ax2.transAxes, fontweight='bold')
    
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.2)
    cbar = fig.colorbar(im, cax=cax)#, orientation='horizontal')
    cbar.ax.set_ylabel('Correlation')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    if display_type=='pdf':
        plt.savefig('corr_mat.pdf', dpi=600, bbox_inches='tight', pad_inches=0.05)
    elif display_type=='png':
        plt.savefig('corr_mat.png', bbox_inches='tight', pad_inches=0.05)
    elif display_type=='tiff':
        plt.savefig('corr_mat.tiff', bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()
