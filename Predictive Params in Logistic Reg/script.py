# Import Relevant Packages
import pandas as pd
import statsmodels.formula.api as sm
import numpy as np
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)

# Import Data
fraud = pd.read_csv("card_transdata.csv")

# Summary Statistics
fraud.describe()

# Build Model Pipeline
features = list(fraud.iloc[:,0:7].columns)

cont_feat = features[:3]
bin_feat = features[3:]

normalize = ColumnTransformer([
    ('cont', StandardScaler(), cont_feat),
    ('binary','passthrough',bin_feat)
    ])

pipeline = Pipeline([
    ('normalize',normalize),
    ('logit',LogisticRegression())
    ])

# Fit Pipeline
model = pipeline.fit(fraud[features],fraud['fraud'])

# Final Model
final_mod = model._final_estimator

# Marginal Effects Function
def logit_margeff(model, X, kind='probability'):
    
    coef = model.coef_
    intercept = model.intercept_
    
    if kind == 'probability':
        
        logodds = intercept+np.dot(X,coef.T)
    
        marg_effects=[]
        for i in range(coef.size):
            marg_eff = np.mean(coef[0,i]*np.exp(-logodds)/(1+np.exp(-logodds))**2).round(3)
            marg_effects.append(marg_eff)
    
    elif kind == "odds":
        
        marg_effects=[]
        for i in range(coef.size):
            marg_eff = (np.exp(coef[0,i])).round(3)
            marg_effects.append(marg_eff)
        
    return marg_effects

# Marginal Effects
logit_margeff(final_mod, fraud[features], kind='probability')
logit_margeff(final_mod, fraud[features], kind='odds')


##############################
## Compute with Statsmodels ##
##############################

logit = sm.logit('''fraud~distance_from_home+distance_from_last_transaction+
                   ratio_to_median_purchase_price+repeat_retailer+used_chip+
                   used_pin_number+online_order'''
                   , data=fraud).fit()

logit.summary()

logit.get_margeff(at='overall').summary()

logit.get_margeff(at='mean').summary()


##################
# Backup Sklearn #
##################

# Z-Score
for x in ['distance_from_home','distance_from_last_transaction',
                          'ratio_to_median_purchase_price']:
    
    fraud[x]=(fraud[x] - fraud[x].mean())/fraud[x].std()
    
# Features and Target
X = fraud.drop('fraud',axis=1).to_numpy()
y = fraud['fraud'].copy()

# Model
log_reg = LogisticRegression()
log_reg.fit(X,y)

# Marginal Effects
logit_margeff(log_reg, X, kind='probability')
logit_margeff(log_reg, X, kind='odds')


#################
#  Shap Values  #
#################

import shap

X_summary = shap.kmeans(X,100)
explainer = shap.KernelExplainer(log_reg.predict_proba, X_summary)

index = np.random.choice(X.shape[0], 10000, replace=False)
X_shap = pd.DataFrame(X[index],columns=features)

repeat_retailer_index = X_shap[X_shap['repeat_retailer']==1].index
used_chip_index = X_shap[X_shap['used_chip']==1].index
used_pin_number_index = X_shap[X_shap['used_pin_number']==1].index
online_order_index = X_shap[X_shap['online_order']==1].index

shap_values = explainer.shap_values(X_shap)

shap_df = pd.DataFrame(shap_values[1],columns=features)
shap_sum = shap_df.sum(axis=1)
pred_probas = pd.Series(log_reg.predict_proba(X[index])[:,1])

from scipy.interpolate import interp1d

shap_sort = shap_sum.sort_values()
pred_probas_sort = pred_probas[shap_sort.index]

interpolate = interp1d(shap_sort, pred_probas_sort,bounds_error=False,fill_value=(0,1))

margins = shap_df.apply(lambda x: shap_sum - x).apply(interpolate).apply(lambda x: pred_probas - x)

margins.mean()
margins.iloc[repeat_retailer_index,:].mean()
margins.iloc[used_chip_index,:].mean()
margins.iloc[used_pin_number_index,:].mean()
margins.iloc[online_order_index,:].mean()



###############################################################################

################
#### Visual ####
################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

x = np.arange(-0.25,1.25,0.001)

z =1/(1+np.exp(-((x-0.5)/0.1)))
y=x

dz = (1/0.1)*(np.exp(-((x-0.5)/0.1))/(1+np.exp(-((x-0.5)/0.1)))**2)

bcfont = {'fontname':'Big Caslon'}
font = font_manager.FontProperties(family='Big Caslon',size=16)

fig, ax = plt.subplots(figsize=(15,10),dpi=1000)
ax.plot(x,y,color='black',label='Linear Probability Model',zorder=1)
ax.plot(x,z,color='#b30000',label='Logit Model',zorder=2)
ax.legend(prop=font)
ax.axhline(y=0,color='black',linestyle='--')
ax.axhline(y=1,color='black',linestyle='--')
#ax.grid(True)
ax.set_ylabel('Predicted Probability',size=15,**bcfont)
ax.set_xlabel('X',size=15,**bcfont)
ax.set_yticks([0,1])
ax.tick_params(axis='both',which='both',labelbottom=False,bottom=False,top=False)
ax.set_title("Linear Probability vs. Logit Model",size=20,**bcfont)
fig.gca().spines['top'].set_visible(False)
fig.gca().spines['right'].set_visible(False)
fig.gca().spines['bottom'].set_visible(False)
fig.gca().spines['left'].set_visible(False)
ax.axhspan(0, -0.25,alpha=0.2,color='black')
ax.axhspan(1, 1.25,alpha=0.2,color='black')
plt.show()

fig, ax = plt.subplots(figsize=(15,10),dpi=1000)
ax.plot(x,dz,color='black',label='Logistic PDF',zorder=1,linewidth=3)
ax.plot(x,z,color='#b30000',label='Logitistc CDF',zorder=2,linewidth=3)
#ax.legend(prop=font)
ax.axhline(y=0,color='black',linestyle='--')
ax.axhline(y=1,color='black',linestyle='--')
#ax.grid(True)
ax.set_ylabel('',size=15,**bcfont)
ax.set_xlabel('',size=15,**bcfont)
ax.set_yticks([])
ax.tick_params(axis='both',which='both',labelbottom=False,bottom=False,top=False)
#ax.set_title("Logistic Distribution",size=20,**bcfont)
fig.gca().spines['top'].set_visible(False)
fig.gca().spines['right'].set_visible(False)
fig.gca().spines['bottom'].set_visible(False)
fig.gca().spines['left'].set_visible(False)
plt.show()

###############################################################################



