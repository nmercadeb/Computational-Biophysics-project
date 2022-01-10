# -*- coding: utf-8 -*-
"""
COMBIO PROJECT: DATA ANALYSIS PART TWO
Nuria Mercade & Marta Alcalde
"""
# Libraries
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import scipy.stats       as stats
import statsmodels.api   as sm
import seaborn           as sns
import scipy.interpolate
from statsmodels.nonparametric.smoothers_lowess import lowess as  sm_lowess
import matplotlib.patches as mpatches

# Read Excel
file = "C:/Users/Marta/Dropbox/COMBIO/dades.xlsx"
#file = "/Users/nmercade/Desktop/dades.xlsx"
# 21-07-21.AT2(648)AE
data_dirty   = pd.read_excel(file,header = 0, usecols = "A,B,H:AD")
data         = data_dirty.drop([0,1,118,219,342,445])
title = ["Moving time (%)", \
         "Average nose speed (cm/s) during video", \
         "Average mouse speed (cm/s) during video", \
         "Average tail speed (cm/s) during video", \
         "Average pawE speed (cm/s) during video", \
         "Average pawD speed (cm/s) during video",\
         "Average pawe speed (cm/s) during video", \
         "Average pawd speed (cm/s) during video", \
         "Average mouse speed (cm/s)", \
         "Distance between front paws (cm)", \
         "Distance between hind legs (cm)", \
         "Mouse size (cm)", \
         "Time with tail curved to contralateral side (%)", \
         "Time wuth tail curved to ipsilateral side (%)", \
         "Time with straight tail (%)", \
         "Time with body curved to contralateral side (%)",\
         "Time with body curved to ipsilateral side (%)", \
         "Time with the body straight (%)", \
         "Turns when moving to contralateral side (%)", \
         "Turns when moving to ipsilateral side (%)", \
         "Walking straight (%)" ]
title1 = ["Moving time (%)", \
         "Average speed (cm/s)", \
         "Average speed (cm/s)", \
         "Average speed (cm/s)", \
         "Average speed (cm/s)", \
         "Average speed (cm/s)",\
         "Average speed (cm/s)", \
         "Average speed (cm/s)", \
         "Average speed (cm/s)", \
         "Distance (cm)", \
         "Distance (cm)", \
         "Mouse size (cm)", \
         "Time (%)", \
         "Time (%)", \
         "Time (%)", \
         "Time (%)",\
         "Time (%)", \
         "Time (%)", \
         "Turns (%)", \
         "Turns (%)", \
         "Walking straight (%)" ]

# =============================================================================
# TAPE REMOVAL - Pearson correlation (TYPE OF FIGURE 1 - 4 SUBPLOTS)
# =============================================================================
param = list(data.head()); param = param[3:];

# Treiem la variable Distance (cm):
param     = param[1:]
trpre     = data.loc[data["Test"] == "TR - PRE"]; 
trpost24h = data.loc[data["Test"] == "TR - POST - 24H"]; 
trpost48h = data.loc[data["Test"] == "TR - POST - 48H"]; 
trpost72h = data.loc[data["Test"] == "TR - POST - 72H"]; 

tr  = {0: trpre, 1: trpost24h, 2: trpost48h, 3: trpost72h};
nam = {0: "Previous", 1: "Post 24h", 2: "Post 48h", 3: "Post 72h"};
infpre  = list(set(trpre["% Infarct"]));     infpre  = np.sort(infpre);
infpost = list(set(trpost24h["% Infarct"])); infpost = np.sort(infpost);
infvec  = {0: infpre, 1: infpost, 2: infpost, 3: infpost};

def smooth(x, y, xgrid,fracid):
    samples = np.random.choice(len(x), 50, replace=True)
    y_s  = y[samples]
    x_s  = x[samples]
    y_sm = sm_lowess(y_s,x_s, frac = fracid, it=10,
                     return_sorted = False)
    # regularly sample it onto the grid
    y_grid = scipy.interpolate.interp1d(x_s, y_sm, 
                                        fill_value='extrapolate')(xgrid)
    return y_grid

import warnings
warnings.filterwarnings("ignore")

for k,i in enumerate(param):
    fig = plt.figure(figsize = (18,15)) 
    fig.suptitle('{}'.format(title[k]), fontsize=18, fontweight='bold')
 
    for j in range(4):
        trj = tr[j];
        r   = stats.pearsonr(trj["% Infarct"],trj[i].dropna());
        if r[1] > 0.1:
            col = 'red'
        if r[1] <= 0.1 and r[1] >= 0.05:
            col = 'orange'
        if r[1] <= 0.05:
            col = 'lightskyblue'
            
        # Scatter
        plt.subplot(2,2,j+1)
        plt.scatter(trj["% Infarct"], trj[i].dropna(), color = col);
        plt.xlabel("% Infarct", fontsize = 15);
        plt.ylabel(title1[k],    fontsize = 15);
        plt.tick_params(labelsize = 15);
        plt.title(nam[j], fontsize = 16, fontweight = "bold")
        print('p value for {} in range {} = {}'.format(title[k],j,r[1]))
        # Line mean
        mn  = []; final = [];
        inf = infvec[j];
        for jj in range(len(inf)):
           xx  = pd.DataFrame(trj[i].dropna())
           xx1 = np.mean(xx.loc[trj["% Infarct"] == inf[jj]])
           mn.append(xx1[0])
      
        #plt.plot(inf, final, label = 'c = {:,.2f}'.format(r[0]), linewidth=2.5)
        #plt.legend(loc='best',fontsize=15)
       
        # Error
        y = np.array(mn); x = inf;
        xgrid = np.linspace(x.min(),x.max())
        K = 100
        smooths = np.stack([smooth(x, y, xgrid, 0.7) for k in range(K)]).T
        
        mean = np.nanmean(smooths, axis=1)
        stderr = scipy.stats.sem(smooths, axis=1)
        stderr = np.nanstd(smooths, axis=1, ddof=0)
        # plot it
        plt.fill_between(xgrid, mean-1.96*stderr,mean+1.96*stderr, alpha=0.25, color = col)
        plt.plot(xgrid, mean, color = col,linewidth = 2.5)
        plt.text(49,np.max(trj[i].dropna()),'c = {:,.2f}'.format(r[0]),fontsize = 15)
        plt.plot(x, y, 'k.')
    red_patch    = mpatches.Patch(color = 'red',       label = 'pvalue > 0.1')
    orange_patch = mpatches.Patch(color = 'orange',    label = '0.05 < pvalue < 0.1')
    blue_patch   = mpatches.Patch(color = 'lightblue', label = 'pvalue < 0.05')
    plt.legend(handles = [blue_patch, orange_patch, red_patch], fontsize=15, \
               loc = 'upper center',  bbox_to_anchor = (-0.1, 2.4), ncol = 3)
    plt.tick_params(labelsize = 15)
    
     
    plt.savefig("{}.png".format(k))

# =============================================================================
# TAPE REMOVAL - Pearson correlation (TYPE OF FIGURE 2 - ONE PLOT)
# =============================================================================
# param = list(data.head()); param = param[3:];
# # Treiem la variable Distance (cm):
# param     = param[1:]
# trpre     = data.loc[data["Test"] == "TR - PRE"]; 
# trpost24h = data.loc[data["Test"] == "TR - POST - 24H"]; 
# trpost48h = data.loc[data["Test"] == "TR - POST - 48H"]; 
# trpost72h = data.loc[data["Test"] == "TR - POST - 72H"]; 

# tr  = {0: trpre, 1: trpost24h, 2: trpost48h, 3: trpost72h};
# nam = {0: "Previous", 1: "Post 24h", 2: "Post 48h", 3: "Post 72h"};
# infpre  = list(set(trpre["% Infarct"]));     infpre  = np.sort(infpre);
# infpost = list(set(trpost24h["% Infarct"])); infpost = np.sort(infpost);
# colors  = ['yellowgreen','seagreen','darkturquoise','lightblue']
# infvec  = {0: infpre, 1: infpost, 2: infpost, 3: infpost};
# import warnings
# warnings.filterwarnings("ignore")

# for k,i in enumerate(param):
#     fig = plt.figure(figsize = (18,15)) 
#     plt.title('TR: {}'.format(title[k]), fontsize=18, fontweight='bold')
    
#     for j in range(4):
#         trj = tr[j];
#         r = stats.pearsonr(trj["% Infarct"],trj[i].dropna());
#         
        
#         # Scatter
#         plt.scatter(trj["% Infarct"], trj[i].dropna(),color = colors[j])
#         plt.xlabel("% Infarct", fontsize = 15);
#         plt.ylabel(title[k],    fontsize = 15);
#         plt.tick_params(labelsize = 15);
          
#         # Line mean
#         mn = []; final = []; inf = infvec[j];
#         for jj in range(len(inf)):
#              xx  = pd.DataFrame(trj[i].dropna())
#              xx1 = np.mean(xx.loc[trj["% Infarct"] == inf[jj]])
#              mn.append(xx1[0])
           
#         # Error
#         y = np.array(mn); x = inf;
#         xgrid = np.linspace(x.min(),x.max())
#         K = 100
#         smooths = np.stack([smooth(x, y, xgrid, 0.7) for k in range(K)]).T
           
#         mean = np.nanmean(smooths, axis=1)
#         stderr = scipy.stats.sem(smooths, axis=1)
#         stderr = np.nanstd(smooths, axis=1, ddof=0)
#         # plot it
#         plt.fill_between(xgrid, mean-1.96*stderr, mean+1.96*stderr, alpha=0.25,color = colors[j])
#         plt.plot(xgrid, mean, color = colors[j], label = '{}; c = {:,.2f}'.format(nam[j],r[0]),linewidth = 2.5)
#         plt.plot(x, y, 'k.')
#         plt.legend( loc = 'best', fontsize = 15)
            
#        # plt.plot(inf, final, color = colors[j], label = '{}; c={:,.2f}'.format(nam[j],r[0]),linewidth=2.5)
#        # plt.legend(loc = 'best',fontsize=15)

#     plt.savefig("TR2{}.png".format(k))
