import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def logit_margeff(
    model: LogisticRegression, X_features: pd.DataFrame, kind: str = "probability"
) -> pd.DataFrame:
    """
    Calculate the marginal effects of a logistic regression model.

    Parameters:
    model (LogisticRegression): The trained logistic regression model.
    X_features (pd.DataFrame): The input features used for prediction.
    kind (str): The type of marginal effects to calculate. Can be "probability" or "odds". Default is "probability".

    Returns:
    pd.DataFrame: A DataFrame containing the features and their corresponding marginal effects.
    """
    coef = model.coef_
    intercept = model.intercept_

    if kind == "probability":
        logodds = intercept + X_features @ coef.T

        marg_effects = []
        for i in range(coef.size):
            marg_eff = np.mean(
                coef[0, i] * np.exp(-logodds) / (1 + np.exp(-logodds)) ** 2
            ).round(3)
            marg_effects.append(marg_eff)

    elif kind == "odds":
        marg_effects = []
        for i in range(coef.size):
            marg_eff = (np.exp(coef[0, i])).round(3)
            marg_effects.append(marg_eff)

    else:
        raise ValueError("kind must be either 'probability' or 'odds'")

    marginal_effects = {}
    marginal_effects["features"] = X_features.columns
    marginal_effects[f"marginal_effects_{kind}"] = marg_effects

    df = pd.DataFrame(marginal_effects)

    return df
