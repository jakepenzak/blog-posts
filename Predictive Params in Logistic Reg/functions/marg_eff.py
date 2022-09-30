import numpy as np

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

