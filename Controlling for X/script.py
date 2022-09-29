# Frisch-Waugh-Lovell Theorem Medium Post

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

sns.set_theme(style="darkgrid")

### Data Generating Process
df = pd.DataFrame()
n = 10000

# Covariates
df['pareduc'] = np.random.normal(loc=14,scale=3,size=n).round()
df['HHinc'] = skewnorm.rvs(5,loc=3,scale=4,size=n).round()
df['IQ'] = np.random.normal(100,10,size=n).round()

# Childhood Monthly Reading
df['read'] = (-25+
              0.3*df['pareduc']+
              2*df['HHinc']+
              0.2*df['IQ']+
              np.random.normal(0,2,size=n)).round()

df = df[(df['read']>=0) & (df['read']<=30)]
                           
# Education Attainment
df['educ'] = (-15+
              0.2*df['read']+
              0.1*df['pareduc']+
              1*df['HHinc']+
              0.2*df['IQ']+
              df['read']/15*np.random.normal(0,2,size=len(df)).round())

# Plot dataset
fig, ax = plt.subplots(3,2,figsize=(12,12))
sns.histplot(df.HHinc,color='b',ax=ax[0,0],bins=15,stat='proportion',kde=True)
sns.histplot(df.IQ,color='m',ax=ax[0,1],bins=20,stat='proportion',kde=True)
sns.histplot(df.pareduc,color='black',ax=ax[1,0],bins=20,stat='proportion',kde=True)
sns.histplot(df.read,color='r',ax=ax[1,1],bins=30,stat='proportion',kde=True)
sns.histplot(df.educ,color='g',ax=ax[2,0],bins=30,stat='proportion',kde=True)
sns.regplot(data=df,x='read',y='educ',color='y',truncate=False,ax=ax[2,1])
plt.show()

import statsmodels.formula.api as sm

## Regression Analysis

# Naive Regression
naive = sm.ols('educ~read',data=df).fit(cov_type="HC3")

# Multiple Regression
multiple = sm.ols('educ~read+pareduc+HHinc+IQ',data=df).fit(cov_type='HC3')

# FWL Theorem
read = sm.ols('read~pareduc+HHinc+IQ',data=df).fit(cov_type='HC3')
df['read_star']=read.resid

educ = sm.ols('educ~pareduc+HHinc+IQ',data=df).fit(cov_type='HC3')
df['educ_star']=educ.resid

FWL = sm.ols("educ_star ~ read_star",data=df).fit(cov_type='HC3')


naive.summary()
multiple.summary()

# Plot both relationships
sns.jointplot(data=df,x='read',y='educ',kind='reg',color='g',truncate=False,
              height=8)
sns.jointplot(data=df,x='read_star',y='educ_star',kind='reg',color='r',
              truncate=False,height=8)


fig, ax = plt.subplots(1,2,figsize=(15,5),dpi=1000)
ax[0].set_title('Naive Regression',fontsize=17)
ax[1].set_title('FWL Regression',fontsize=17)
sns.regplot(data=df,x='read',y='educ',color='y',truncate=False,ax=ax[0])
sns.regplot(data=df,x='read_star',y='educ_star',color='y',truncate=False,
            ax=ax[1])
plt.show()

# Save Table

from stargazer.stargazer import Stargazer

file = open('table.html','w')

order = ['read','read_star','HHinc','pareduc','IQ','Intercept']
columns = ['Naive OLS','Multiple OLS','FWL']
rename = {'read':'Read (Days/Month)','read_star':'Read*','hhincome':'HH Income',
          'pareduc':"Avg. Parents Education (Yrs)"}

regtable = Stargazer([naive, multiple, FWL])
regtable.covariate_order(order)
regtable.custom_columns(columns,[1,1,1])
regtable.rename_covariates(rename)
regtable.show_degrees_of_freedom(True)
regtable.title('The Simulated Effect of Childhood Reading on Educational Attainment')

file.write(regtable.render_html())
file.close()
