import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    ## Relevant Imports

    import marimo as mo
    import pandas as pd
    import statsmodels.formula.api as sm
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from IPython.display import display, HTML
    import os

    try:
        os.chdir("assets/articles/notebooks")
    except:
        pass

    pd.set_option("display.max_columns", None)
    return (
        ColumnTransformer,
        HTML,
        LogisticRegression,
        Pipeline,
        StandardScaler,
        display,
        mo,
        np,
        os,
        pd,
        plt,
        sm,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Predictive Parameters in a Logistic Regression: Making Sense of it All

        <center> **Acquire a robust understanding of logit model parameters beyond the canonical odds ratio interpretation** </center>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Introduction

        Logistic regressions, also referred to as a logit models, are powerful alternatives to linear regressions that allow one to model a binary outcome (i.e., 0 or 1) and provide notably accurate predictions on the probability of said outcome occurring given an observation. The parameter estimates within logit models can provide insights into how different explanatory variables, or features, contribute to the model predictions. Many readers are likely familiar with interpreting logit model parameters in terms of **odds ratios** (if not don't worry, I briefly cover this below). Nevertheless, the interpretation of these parameters in terms of **probabilities** is not as straightforward, but a robust understanding of how to interpret these parameters can provide an immense amount of intuition into explaining the model's underlying predictive behavior.

        > Making a prediction is extremely powerful, but intuitively explaining the predictive components of a model in real world terms can take your project analysis to the next level.

        By the end of this article you will see logistic regression in a new light and gain an understanding of how to explain the model parameters with a staggering amount of intuition. This article assumes a brief underlying knowledge of logit models and thus directs focus more intently on interpreting the model parameters in a comprehensible manner. Nevertheless, we will first briefly discuss the theory behind logit models. We will then give an in-depth discussion into how to interpret the model parameters as marginal effects. Lastly, we will cover a practical example predicting fraudulent credit card transactions utilizing the following [Kaggle dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud?resource=download).

        ## Logistic Regression Crash Course

        [Logit Models](https://en.wikipedia.org/wiki/Logistic_regression) belong to a more broad family of [generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLMs) that, in brief, allow for flexible fitting of linear models when the outcome of interest follows a different underlying distribution than gaussian and relates the linear model to the outcome of interest via a link function. The canonical linear regression is a special case where the link function is the identity function. In the binary outcome case, a linear regression, which is referred to as linear probability model, can provide predictions that are less than 0 or greater than 1 (See Figure 1). This clearly poses issues as probabilities are naturally bounded between 0 and 1. However, GLM's provide a convenient framework to solve this problem!

        The logit model is a specific case that allows for the modeling of binary outcomes that follow a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution). The logit model is particularly attractive because the link function used (logit function) is bounded between 0 and 1. And consequently, all model predictions are bounded between 0 and 1, as anticipated in the probability space. Figure 1 below provides a nice visual comparison between the model fits of linear probability model and logistic regression in a bivariate case.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    def lin_prob_vs_logistic_viz():
        x = np.arange(-0.25, 1.25, 0.001)
        z = 1 / (1 + np.exp(-((x - 0.5) / 0.1)))
        y = x
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(x, y, color="black", label="Linear Probability Model", zorder=1)
        ax.plot(x, z, color="#b30000", label="Logit Model", zorder=2)
        ax.legend()
        ax.axhline(y=0, color="black", linestyle="--")
        ax.axhline(y=1, color="black", linestyle="--")
        # ax.grid(True)
        ax.set_ylabel("Predicted Probability", size=15)
        ax.set_xlabel("X \n \n Figure 1", size=15)
        ax.set_yticks([0, 1])
        ax.tick_params(
            axis="both", which="both", labelbottom=False, bottom=False, top=False
        )
        ax.set_title("Linear Probability vs. Logit Model", size=20)
        fig.gca().spines["top"].set_visible(False)
        fig.gca().spines["right"].set_visible(False)
        fig.gca().spines["bottom"].set_visible(False)
        fig.gca().spines["left"].set_visible(False)
        ax.axhspan(0, -0.25, alpha=0.2, color="black")
        ax.axhspan(1, 1.25, alpha=0.2, color="black")
        plt.savefig("data/lin_prob_vs_logistic_viz.webp", format="webp", dpi=200)

    lin_prob_vs_logistic_viz()
    mo.image("data/lin_prob_vs_logistic_viz.webp").center()
    return (lin_prob_vs_logistic_viz,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Mathematically, the logit model is characterized by:

        $$
        \begin{equation}
        p(X_i)=\frac{1}{1+e^{-(X_i\beta)}}, \space \space \space y \sim \text{Bernoulli}(p(X_i))
        \tag{1}
        \end{equation}
        $$

        Where $X$ denotes the matrix of observed explanatory variables, or features, in the model and $p(X)$ denotes the probability of $y$ taking on a value of 1. Given this model setup with y distributed Bernoulli, the goal of logit model estimation is to maximize the following likelihood function, which is our joint distribution:

        $$
        \begin{equation}
        L=\prod_{i:y_i=1}p(X_i)\prod_{i:y_i=0}1-p(X_i)
        \tag{2}
        \end{equation}
        $$

        In simple terms, our optimization problem seeks to choose the parameters (i.e., $β$ ) in (1) that will maximize (2). Note that (2) will be maximized when the estimated probability is close to 1 for individuals with $y$ = 1 and close to 0 for individuals with $y$ = 0. To do so, one can take the log of the likelihood function to obtain the log-likelihood and solve this problem using gradient descent or related algorithms. For more details, the [wiki](https://en.wikipedia.org/wiki/Logistic_regression) page on logistic regression provides a nice in-depth treatment to logit model estimation.

        ## Interpreting Logit Parameters as Marginal Effects

        A marginal effect can be thought of as the average (or marginal) effect on the outcome (or target) variable resulting from a change in the explanatory variable (or feature) of interest. This can similarly be conceptualized as follows: At the average (or marginal) observation/individual, what is the effect of changing an explanatory variable on the outcome. In our case with a binary variable, this would be akin to estimating the average effect of changing an explanatory variable on the probability of observing the outcome.

        > **Caution:** Marginal effects must be interpreted only as an association and not as a causal relationship. Causality requires additional identifying assumptions.

        By recognizing that the marginal effect is simply a rate of change, one may immediately notice that this boils down to a taking a derivative with respect to the explanatory variable. We will first start with the simple linear regression case to make things easy. Suppose we have the following linear regression:

        $$
        \begin{equation}
        y_i= X_i \mathbf{\beta} + \epsilon_i =\beta_0+\beta_1x_{1i}+\beta_2x_{2i}+ \dots + \beta_kx_{ki}+\epsilon_i 
        \tag{3}
        \end{equation}
        $$

        In order to find the marginal effect of any variable, we can take the derivative with respect to the $x$ of interest in (3). This derivative for any $x^*$ is simply:

        $$
        \begin{equation}
        \frac{\partial y_i}{\partial x^*}=\beta^*
        \tag{4}
        \end{equation}
        $$

        Note, in this case, we have a constant marginal effect, which makes sense because a linear regression is a linear projection of y onto X. The marginal effect can be interpreted as follows:

        > **Interpretation**: On average, a one unit increase in $x^*$ is associated with a $β^*$ change in $y$.

        Now the careful reader may notice that this derivative is not nearly as trivial for logit models (See below for a discussion into log-odds and odds ratios). Consider the logistic model outlined in eq. (1). The derivative with respect to any x* can be solved for using the chain and quotient rules. We can thus find the marginal effect of $x^*$ on the probability of $y$ occurring as follows:

        $$
        \begin{equation}
        \frac{\partial p(X_i)}{\partial x^*}=\beta^* \times \frac{e^{-X_i \mathbf{\beta} }}{(1+e^{-X_i \mathbf{\beta}})^2}
        \tag{5}
        \end{equation}
        $$

        Here we can see that the marginal effect is now a function of the values of $X$. This again makes sense as the logit function is non-linear (See Figure 1). This gives us the power to evaluate the marginal effects at any combination of $X$. However, if we want to summarize the overall marginal effects we are left with two options:

        1. **Compute the average marginal effect** - This entails computing the marginal effect using (5) for every observation and then computing the mean value

        2. **Compute the marginal effect at the average** - This entails plugging in the mean values of all of the explanatory variables into (5) and computing the marginal effect

        There is not an immediately apparent benefit of one over the other and both provide different interpretations under different contexts. However, the average marginal effect provides the cleanest interpretation, and thus will be the one we work with for the remainder of this post.

        > Note that all calculations can easily be extended to compute the marginal effects not only at the average values of the explanatory variables, but at any combination of values. I will leave this for the interested reader and the code provided in the next section can be readily augmented to do so (i.e., plug in the values of each variable you are interested into (5) to obtain the marginal effect at that observation). This can provide **very powerful** insights into how the predictive parameter marginal effects vary by certain types of individuals/observations!


        Nevertheless, the interpretation of the average marginal effect in a logit model is as follows:

        > **Interpretation:** On average, a one unit increase in $x^*$ is associated with a {computed value} percentage point change in the probability of $y$ occurring.

        ## Log odds, Odds, and the Odds Ratio

        Before we provide a numerical example of this in action, it is important to discuss the relationship between logit models, log odds, odds, and the odds ratios. It is quite common that logistic regression results are interpreted in terms of odds, and this is because, after some algebra, we can rewrite (1) as:

        $$
        \begin{equation}
        \ln \left( \frac{p(X_i)}{1-p(X_i)} \right) = X_i \beta
        \tag{6}
        \end{equation}
        $$

        Where the left hand side is in log-odds. Thus, a logistic regression has a constant marginal effect in terms of log odds, where:

        $$
        \begin{equation}
        \frac{\partial \ln \left( \frac{p(X_i)}{1-p(X_i)} \right)}{\partial x^*} = \beta^*
        \tag{7}
        \end{equation}
        $$

        However, marginal effects in terms of log-odds is extremely removed from any intuition. Thus, one can solve for the model in terms of odds by taking the exponential of (6):

        $$
        \begin{equation}
        \frac{p(X_i)}{1-p(X_i)} = e^{X_i \beta}
        \tag{8}
        \end{equation}
        $$

        It is then commonplace that the logistic regression parameters are interpreted in terms of odds by computing the odds ratios where, using (8) and incrementing $x^*$ by 1, we obtain:

        $$
        \begin{equation}
        \text{Odds Ratio} = \frac{e^{X_i \beta + \beta^*}}{e^{X_i \beta}} = e^{\beta^*}
        \tag{9}
        \end{equation}
        $$

        The interpretation is as follows:

        > **Interpretation:** On average, a one unit increase in $x^*$ is associated with multiplying the odds of $y$ occurring by $\beta^*$.

        In my opinion, the interpretation of these are not always as clearcut as a probability interpretation unless one has exposure to and works with log-odds, odds, and odds ratios regularly. Nevertheless, (7–9) can provide insights into the marginal effects of $x^*$ on the log-odds, odds, and odds ratio, respectively.

        ## OPTIONAL: Nonlinearities & Interactions

        Suppose we had the two following beliefs: $x^*$ has a quadratic relationship with $y$ and we believe the effect to differ by gender. We can augment our logit model to include two additional engineered features as follows:

        $$
        \begin{equation}
        p(X_i)=\frac{1}{1+e^{-(\beta_0+\beta^*x^*_i+\beta^{**}x^{*2}_i+\beta^{***}x_i^*\text{male}_i + \dots + \beta_k x_{ki})}}
        \tag{10}
        \end{equation}
        $$

        where we include the squared term of $x^*$ and interact $x^*$ with a dummy variable for if that individual is male or not. Thus, our interpretation of the marginal effect will now be slightly less nuanced.

        > Note that whenever we include an interaction term we must take care to include each variable un-interacted first in the model (i.e., also include the dummy male alone as well). Otherwise, the interaction term will eat the raw effect of gender on $y$ when in reality the interaction term may be redundant.

        Now differentiating (10) with respect to $x^*$ we obtain:

        $$
        \begin{equation}
        \frac{\partial p(X_i)}{\partial x^*}=(\beta^*+2\beta^{**}x_i^*+\beta^{***}\text{male}_i) \times \frac{e^{-X_i\beta}}{(1+e^{-X_i\beta})^2}
        \tag{11}
        \end{equation}
        $$

        We can now see that, due to the nonlinearities, the marginal effect will vary further depending on the value of $x^*$ and whether that individual is male or female. This can allow us to then compute average marginal effects for males versus females by computing (11) for each male and female then taking the average for each. We can similarly compute the odds ratio as done in (9) after solving (10) in terms of odds. These examples will be left for the interested reader, and what we have covered so far should be sufficient to compute these.

        ## The Marginal Effects in Predicting Credit Card Fraud

        To demonstrate a concrete example of what we discussed above, we will utilize the following [Kaggle dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud?resource=download) on credit card transactions with the intent of building a model to predict whether a transaction is fraudulent. The dataset has the following variables on each transaction:

        First, let's import the data and generate summary statistics:
        """
    )
    return


@app.cell(hide_code=True)
def _(pd):
    fraud = pd.read_csv("data/card_transdata.csv")

    fraud.describe().round(2)
    return (fraud,)


@app.cell
def _(mo):
    mo.md(
        r"""We will now build a logistic regression model using scikit-learn. Suppose we have already gone through the proper steps in training and validating the model and have determined the appropriate model. Our final model is as follows:"""
    )
    return


@app.cell
def _(ColumnTransformer, LogisticRegression, Pipeline, StandardScaler, fraud):
    features = list(fraud.iloc[:, 0:7].columns)

    cont_feat = features[:3]
    bin_feat = features[3:]

    normalize = ColumnTransformer(
        [("cont", StandardScaler(), cont_feat), ("binary", "passthrough", bin_feat)]
    )

    pipeline = Pipeline([("normalize", normalize), ("logit", LogisticRegression())])

    # Fit Pipeline
    model = pipeline.fit(fraud[features], fraud["fraud"])

    # Final Model
    final_mod = model._final_estimator
    return bin_feat, cont_feat, features, final_mod, model, normalize, pipeline


@app.cell
def _(mo):
    mo.md(
        """We have built our logit model to predict if a credit card transaction is fraudulent. Now let's pivot into explaining the model parameters to understand the inner workings of the model and the subsequent role each feature plays in driving predictions. We will define a function to compute the marginal effects of the logistic regression both in terms of probabilities and odds:"""
    )
    return


@app.cell
def _(LogisticRegression, np, pd):
    def logit_margeff(
        model: LogisticRegression, X_features: pd.DataFrame, kind: str = "probability"
    ) -> pd.DataFrame:
        """
        Calculate the marginal effects of a logistic regression model.

        Parameters
        ----------
        model
            The trained logistic regression model.
        X_features
            The input features used for prediction.
        kind
            The type of marginal effects to calculate. Can be "probability" or "odds".
            Default is "probability".

        Returns
        --------
        pd.DataFrame
            A DataFrame containing the features and their corresponding marginal effects.
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

    return (logit_margeff,)


@app.cell
def _(mo):
    mo.md(
        r"""
        > Note that line 14 is the average marginal effect calculated using (5) and line 21 is the odds ratio calculated using (9).

        After we have defined this function all we have to do is feed in the logit model we have built and the matrix of features. Let's first interpret the output in terms of probabilities:
        """
    )
    return


@app.cell
def _(features, final_mod, fraud, logit_margeff):
    logit_margeff(final_mod, fraud[features], kind="probability")
    return


@app.cell
def _(mo):
    mo.md(
        """
        Recall we have standardized all continuous features and thus a one unit increase corresponds to a one standard deviation increase. We will interpret the estimated average marginal effects for one continuous feature, distance_from_home, and one binary feature, used_pin_number.

        > **Interpretation (distance_from_home):** On average, a one standard deviation (65.391) increase in the distance the transaction occurred from the cardholders home address is associated with a 2.4 percentage point increase in the probability that the transaction is fraudulent.
        >
        > **Interpretation (used_pin_number):** On average, a credit card transaction that included the use of a pin number is associated with a 32.3 percentage point decrease in the probability that the transaction is fraudulent.

        Now, in terms of odds:
        """
    )
    return


@app.cell
def _(features, final_mod, fraud, logit_margeff):
    logit_margeff(final_mod, fraud[features], kind="odds")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Conclusion

        I hope this post has helped you learn how to extract **meaningful insights** from logit model parameters. It is clear that marginal effect interpretations in terms of probabilities provide an immense amount of intuition and explainability of the predictive mechanics under a logit model framework. Generally speaking, these parameters explain how the model makes predictions as well as explain associations between the target and features. However, under additional identifying assumptions, we can make more powerful statements towards interpreting model parameters as a causal relationship between certain features and targets. I hope this post has increased your knowledge and appreciation for logistic regressions!

        ## References

        Dataset available on Kaggle: [Credit Card Fraud](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud?resource=download) (License: CC0: Public Domain)


        <div style="text-align: center; font-size: 24px;">❖❖❖</div>

        <center>
        Access all the code via this Marimo Notebook or my [GitHub Repo](https://github.com/jakepenzak/blog-posts)

        I appreciate you reading my post! My posts primarily explore real-world and theoretical applications of econometric and statistical/machine learning techniques, but also whatever I am currently interested in or learning 😁. At the end of the day, I write to learn! I hope to make complex topics slightly more accessible to all.
        </center>
        """
    )
    return


if __name__ == "__main__":
    app.run()
