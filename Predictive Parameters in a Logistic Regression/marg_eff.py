import numpy as np
import pandas as pd


# Marginal Effects Function
def logit_margeff(model, X, features, kind="probability"):
    coef = model.coef_
    intercept = model.intercept_

    if kind == "probability":
        logodds = intercept + X @ coef.T

        marg_effects = []
        for i in range(coef.size):
            marg_eff = np.mean(
                coef[0, i] * np.exp(-logodds) / (1 + np.exp(-logodds)) ** 2
            ).round(3)[0]
            marg_effects.append(marg_eff)

    elif kind == "odds":
        marg_effects = []
        for i in range(coef.size):
            marg_eff = (np.exp(coef[0, i])).round(3)
            marg_effects.append(marg_eff)

    marginal_effects = {}
    marginal_effects["features"] = features
    marginal_effects[f"marginal_effects_{kind}"] = marg_effects

    df = pd.DataFrame(marginal_effects)

    return df
