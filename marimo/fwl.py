import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # Relevant Imports

    import marimo as mo

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import skewnorm
    import statsmodels.formula.api as sm
    from stargazer.stargazer import Stargazer  # noqa: F401
    from IPython.display import display, HTML
    import os

    try:
        os.chdir("assets/articles/notebooks")
    except:
        pass

    sns.set_theme(style="darkgrid")
    return HTML, Stargazer, display, mo, np, os, pd, plt, skewnorm, sm, sns


@app.cell
def _(mo):
    mo.md(
        """
        # Controlling for "X": 
        <center> **Understanding linear regression mechanics via the Frisch-Waugh-Lovell Theorem** </center>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Introduction

        Applied econometrics is generally interested in establishing causality. That is, what is the “treatment” effect of $T$ on some outcome $y$. In a simple bivariate case, we can imagine randomly assigning treatment $T=1$ to some individuals and $T=0$ to others. This can be represented by the following linear regression model:

        $$
        \begin{equation}
        y_i = \beta_0 + \beta_1T_i+\epsilon_i 
        \tag{1}
        \end{equation}
        $$

        If we assume the treatment is truly randomly assigned, then $T$ is independent to the error term or, in the economists jargon, exogenous. Therefore, we can estimate eq. (1) using ordinary least squares (OLS) and interpret the coefficient estimate on $T$ with a causal interpretation - the average treatment effect (ATE):

        $$
        \begin{equation}
        \text{ATE}=\mathbb{E_n}[y(T=1)-y(T=0)]=\mathbb{E}[(\beta_0+\beta_1)-\beta_0]=\beta_1 
        \tag{2}
        \end{equation}
        $$

        However, when we are dealing with non-experimental data, it is almost always the case that the treatment of interest is not independent to the error term or, again in the economists jargon, endogenous. For example, suppose we were interested in identifying the treatment effect of time spent reading books as a child on an individuals future educational attainment. Without any random assignment of time spent reading as a child, estimating a naïve linear regression, as in eq. (1), will fail to capture a large set of additional factors that may drive individual time spent reading books and educational attainment (i.e., socioeconomic status, parents education, underlying ability, other hobbies, etc.). Thus, when dealing with non-experimental data, we must rely on controlling for additional covariates and then make an argument that treatment is now "as good as randomly assigned" to establish causality. This is known as the conditional independence assumption (CIA). In our educational example above, we can reframe eq. (1) as:

        $$
        \begin{equation}
        \text{Education}_i=\beta_0+\beta_1\text{Read}_i+\mathbf{X}_i \Phi + \epsilon_i
        \tag{3}
        \end{equation}
        $$

        where we now control for a set of observed covariates $X$. The key estimate of interest on Read takes on a causal interpretation if and only if the CIA holds. That is, time spent reading is exogenous (i.e., no uncontrolled confounders) conditional on $X$. Equivalently,

        $$
        \begin{equation}
        \text{cov}(\text{Read}_i,\epsilon_i|\mathbf{X}_i)=0
        \tag{4}
        \end{equation}
        $$

        Without the CIA, our coefficient estimates are biased and we are limited in what we can say in terms of causality. Realistically, it is often quite difficult to make the argument for the CIA and, unfortunately, this assumption is not directly testable. In fact, what I have discussed above is a fundamental motivator for an entire field of econometrics that is devoted to establishing, developing, and implementing quasi-experimental research designs to establish causality including, but most definitely not limited to, difference-in-differences, synthetic control, and instrumental variable designs. These quasi-experimental designs seek to exploit exogenous ("as good as random") sources of variation in a treatment of interest $T$ to study the causal effect of $T$ on some outcome(s) $y$. There are some excellent econometric texts that are accessible to those with little to no background in econometrics, including "The Effect" by Nick Huntington-Klein, "Causal Inference: The Mixtape" by Scott Cunningham, or "Causal Inference for the Brave and True" by Matheus Facure Alves.[1][2][3] Joshua Angrist and Jörn-Steffen Pischke provide a deeper dive in "Mostly Harmless Econometrics" for those interested.[4]

        Despite the fact that establishing the CIA is particularly difficult through controlling for covariates alone, there is a substantial theorem in econometrics that provides some very powerful intuition into what it really means to "control" for additional covariates. Ultimately, this not only provides a deeper understanding to the underlying mechanisms of a linear regression, but also how to conceptualize key relationships of interest (i.e., the effect of $T$ on $Y$).

        > Note that I have (intentionally) glossed over some additional causal inference/econometric assumptions, such as Positivity/Common Support & SUTVA/Counterfactual Consistency. In general, the CIA/Ignorability assumption is the most common assumption that needs to be defended. However, it is recommended that the interested reader familiarize themselves with the additional assumptuons. In brief, Positivity ensures we have non-treated households that are
        similar & comparable to treated households to enable counterfactual estimation & SUTVA ensures there is no
        spillover/network type effects (treatment of one individual impacts another).

        ## Frisch-Waugh-Lovell Theorem

        In the 19th century, econometricians Ragnar Frisch and Frederick V. Waugh developed, which was later generalized by Michael C. Lovell, a ~super cool~ theorem (the FWL Theorem) that allows for the estimation of any key parameter(s) in a linear regression where one first "partials out" the effects of the additional covariates.[5][6] First, a quick refresher on linear regression will be helpful.

        A linear regression solves for the best linear predictors for an outcome $y$ given a set of variables $X$, where the fitted values of $y$ are projected onto the space spanned by $X$. In matrix notation, the linear regression model we are interested in is characterized by:

        $$
        \begin{equation}
        y=\mathbf{X} \beta + \epsilon
        \tag{5}
        \end{equation}
        $$

        The objective of a linear regression is to minimize the residual sum of squares (RSS), thus can be solved via the following optimization problem:

        $$
        \begin{equation}
        \min_{\beta}(y-\mathbf{X}\beta)'(y-\mathbf{X}\beta)
        \tag{6}
        \end{equation}
        $$

        Taking the derivative and setting equal to zero, the optimal solution to (6) is:

        $$
        \begin{equation}
        \beta^* = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'y
        \tag{7}
        \end{equation}
        $$

        This is the ordinary least squares (OLS) estimator that is the workhorse behind the scenes when we run a linear regression to obtain the parameter estimates. Now with that refresher out of the way, let's get to what makes the FWL so great.

        Let's return to our example of estimating the educational returns to reading as a child. Suppose we only want to obtain the key parameter of interest in eq. (3); that is, the effect of days per month spent reading as a child on educational attainment. Recall that in order to make a causal statement about our estimate, we must satisfy the CIA. Thus, we can control for a set of additional covariates X and then estimate (3) directly using the OLS estimator derived in (7). However, the FWL Theorem allows us to obtain the exact same key parameter estimate on Read under the following 3-step procedure:

        1. Regress Read onto the set of covariates $X$ only and, similarly, regress Education onto the set of covariates $X$ only

        $$
        \begin{equation}
        \text{Read}_i=\psi_0+\mathbf{X}_i\mathbf{\Psi }+\xi_i
        \tag{8}
        \end{equation}
        $$

        $$
        \begin{equation}
        \text{Education}_i=\omega_0+\mathbf{X}_i\mathbf{\Omega }+\nu_i
        \tag{9}
        \end{equation}
        $$

        2. Store the residuals after estimating (8)+(9) denoted $\text{Read}_i^*$ and $\text{Education}_i^*$

        $$
        \begin{equation}
        \text{Read}_i^*=\text{Read}_i - \widehat{\text{Read}_i}
        \tag{10}
        \end{equation}
        $$

        $$
        \begin{equation}
        \text{Education}_i^*=\text{Education}_i - \widehat{\text{Education}_i}
        \tag{11}
        \end{equation}
        $$

        3. Regress $\text{Education}_i^*$ onto $\text{Read}_i^*$

        $$
        \begin{equation}
        \text{Education}_i^*=\beta_0+\beta^*\text{Read}_i^*+\epsilon
        \tag{12}
        \end{equation}
        $$

        And that's it!

        Intuitively, the FWL theorem partials out the variation in Read (the treatment/variable of interest) and Education (the outcome of interest) that is explained by the additional covariates, and then uses the remaining variation to explain the key relationship of interest. This procedure can be generalized for any number of key variables of interest. For a more formal proof of this theorem, refer to [7]. The FWL theorem has been in the spotlight recently as the theoretical underpinning for debiased/orthogonal machine learning where steps 1 and 2 are conducted using machine learning algorithms rather than OLS. There are very cool developments occurring that are bridging the gaps between econometrics and machine learning, and I hope to have future posts with some cool applications with respect to some of these new methods. However, part 2 of Matheus Facure Alves' "Causal Inference for the Brave and True" is a great place to start.

        Now you may wonder why in the world would you ever go through this process to obtain the exact same key estimate. Well for one, it provides an immense amount of intuition behind the mechanisms in a linear regression. Secondly, it allows you to visualize the remaining variation in your treatment (Read) that is being used to explain the remaining variation in your outcome (Education). Let us look at this in action!

        ## FWL Theorem Application

        In this section, we are going to simulate a highly stylized dataset to provide a simplified numerical example of applying the FWL theorem in answering our empirical question of the educational returns to childhood reading.
        Suppose we hypothesize a set of demographic variables that we determine to be the relevant confounders necessary to satisfy the CIA in eq. (3), and thus allowing us to obtain a causal interpretation for the education returns to childhood reading. Namely, suppose we identify the key confounders to be the average education level of both parents in years ($\text{pareduc}$), household income as a child in tens of thousands of dollars ($\text{HHinc}$), and IQ score ($\text{IQ}$). We will artificially generate our dataset and the following data generating process (DGP) for the confounders as follows:

        $$
        \text{pareduc}_i \sim \mathcal{N}(14, 3) \newline
        \text{HHinc}_i \sim \mathcal{SN}(3,4,5) \newline
        \text{IQ}_i \sim \mathcal{SN}(100, 10) \newline
        $$

        Furthermore, to estimate eq. (3) we must have measures for the key treatment, average number of days in a month they read as a child ($\text{read}$), and the main outcome, their total educational attainment in years ($\text{educ}$). We artificially generate these key variables with gaussian error terms and heteroskedasticity in the education error term as follows:

        $$
        \begin{equation}
        \text{read}_i = -25 + 0.3\times\text{pareduc}_i+2\times\text{HHinc}_i+0.2\times\text{IQ}_i+\epsilon_i^r
        \tag{13}
        \end{equation}
        $$

        $$
        \begin{equation}
        \text{educ}_i = -15 + 0.2 \times \text{read}_i + 0.1\times\text{pareduc}_i+1\times\text{HHinc}_i+0.2\times\text{IQ}_i+\frac{\text{read}_i}{15}\epsilon_i^e
        \tag{14}
        \end{equation}
        $$

        Because we know the true DGP, the true value for the parameter of interest is 0.2. Let's take this DGP to Python and simulate the data:

        > Note that all values in the DGP were, in general, chosen arbitrarily such that the data works nicely for demonstration purposes. However, within the realm of this simulation we can interpret the coefficient on "read" as follows: On average, for each additional day a month that an individual read as child, their educational attainment increased by 0.2 years.

        First, let's generate the data:
        """
    )
    return


@app.cell
def _(np, pd, skewnorm):
    df = pd.DataFrame()
    n = 10000

    # Covariates
    df["pareduc"] = np.random.normal(loc=14, scale=3, size=n).round()
    df["HHinc"] = skewnorm.rvs(5, loc=3, scale=4, size=n).round()
    df["IQ"] = np.random.normal(100, 10, size=n).round()

    # Childhood Monthly Reading
    df["read"] = (
        -25
        + 0.3 * df["pareduc"]
        + 2 * df["HHinc"]
        + 0.2 * df["IQ"]
        + np.random.normal(0, 2, size=n)
    ).round()

    df = df[(df["read"] >= 0) & (df["read"] <= 30)]

    # Education Attainment
    df["educ"] = (
        -15
        + 0.2 * df["read"]
        + 0.1 * df["pareduc"]
        + 1 * df["HHinc"]
        + 0.2 * df["IQ"]
        + df["read"] / 15 * np.random.normal(0, 2, size=len(df)).round()
    )
    return df, n


@app.cell
def _(mo):
    mo.md(r"""The data will look a little something like:""")
    return


@app.cell(hide_code=True)
def _(df, mo, plt, sns):
    def data_hists():
        fig, ax = plt.subplots(3, 2, figsize=(15, 15), dpi=300)
        sns.histplot(
            df.HHinc, color="b", ax=ax[0, 0], bins=15, stat="proportion", kde=True
        )
        sns.histplot(
            df.IQ, color="m", ax=ax[0, 1], bins=20, stat="proportion", kde=True
        )
        sns.histplot(
            df.pareduc, color="black", ax=ax[1, 0], bins=20, stat="proportion", kde=True
        )
        sns.histplot(
            df.read, color="r", ax=ax[1, 1], bins=30, stat="proportion", kde=True
        )
        sns.histplot(
            df.educ, color="g", ax=ax[2, 0], bins=30, stat="proportion", kde=True
        )
        sns.regplot(data=df, x="read", y="educ", color="y", truncate=False, ax=ax[2, 1])
        plt.savefig("data/data_hists.webp", format="webp", dpi=300, bbox_inches='tight')

    data_hists()
    mo.image("data/data_hists.webp").center()
    return (data_hists,)


@app.cell
def _(mo):
    mo.md(r"""The graph in the bottom right provides the scatter plot and naïve regression line of educ on read. This relationship, on the surface, shows a very strong positive relationship between days read a month as a child and educational attainment. However, we know that by construction this is not the true relationship between educ and read because of the common confounding covariates. We can quantify this result and the bias more formally via regression analysis. Let's now go ahead and estimate the naïve regression (i.e., eq. (3) less $X$), the multiple regression with all relevant covariates (i.e., eq. (3)), and the FWL 3 step process (i.e., eqs. (8)-(12)):""")
    return


@app.cell
def _(df, sm):
    ## Regression Analysis

    # Naive Regression
    naive = sm.ols("educ~read", data=df).fit(cov_type="HC3")

    # Multiple Regression
    multiple = sm.ols("educ~read+pareduc+HHinc+IQ", data=df).fit(cov_type="HC3")

    # FWL Theorem
    read = sm.ols("read~pareduc+HHinc+IQ", data=df).fit(cov_type="HC3")
    df["read_star"] = read.resid

    educ = sm.ols("educ~pareduc+HHinc+IQ", data=df).fit(cov_type="HC3")
    df["educ_star"] = educ.resid

    FWL = sm.ols("educ_star ~ read_star", data=df).fit(cov_type="HC3")
    return FWL, educ, multiple, naive, read


@app.cell
def _(mo):
    mo.md(r"""The regression results are:""")
    return


@app.cell(hide_code=True)
def _(FWL, HTML, Stargazer, multiple, naive):
    def prettify_ols_results():
        order = ["read", "read_star", "HHinc", "pareduc", "IQ", "Intercept"]
        columns = ["Naive OLS", "Multiple OLS", "FWL"]
        rename = {
            "read": "Read (Days/Month)",
            "read_star": "Read*",
            "hhincome": "HH Income",
            "pareduc": "Avg. Parents Education (Yrs)",
        }

        regtable = Stargazer([naive, multiple, FWL])
        regtable.covariate_order(order)
        regtable.custom_columns(columns, [1, 1, 1])
        regtable.rename_covariates(rename)
        regtable.show_degrees_of_freedom(False)
        regtable.title(
            "Table 1: The Effect of Childhood Reading on Educational Attainment"
        )

        return regtable

    regtable = prettify_ols_results()
    HTML(f"<center>{regtable.render_html()}</center>")
    return prettify_ols_results, regtable


@app.cell
def _(mo):
    mo.md(
        r"""
        Table 1 above presents the regression output results. Immediately, we can observe that the naïve regression estimate on read is biased upwards due to the confounding variables that are both positively related with educational attainment and childhood reading. When we include the additional covariates in column (2), we get an estimate near the true value of 0.2 as constructed in the DGP. The FWL 3-step process yields the exact same estimate, as expected!

        > A general rule of thumb for signing bias in a regression is the sign of cov(outcome,X) multiplied by the sign of cov(treatment,X). By construction, we have the cov(educ,X)>0 and cov(read,X)>0 and, hence, positive bias.

        So, we have now shown the FWL being used to obtain the same estimate, but the real power in FWL lies in the ability to plot the true relationship. Figure 2 below shows the initial relationship of the naïve regression without factoring in the covariates and then the relationship of the residuals from the FWL process, where the noise is from the stochastic error term in DGP. In this case, the FWL slope is the true relationship! We can see how vastly different the slope estimates are. This is where the true power of the FWL theorem lies! It allows us to visualize the relationship between a treatment and outcome after we partial out the variation that is already explained by the additional covariates.
        """
    )
    return


@app.cell(hide_code=True)
def _(df, mo, plt, sns):
    def fwl_residual_plot():
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_title("Naive Regression", fontsize=17)
        ax[1].set_title("FWL Regression", fontsize=17)
        sns.regplot(data=df, x="read", y="educ", color="y", truncate=False, ax=ax[0])
        sns.regplot(
            data=df, x="read_star", y="educ_star", color="y", truncate=False, ax=ax[1]
        )
        ax[1].set_xlabel("$read^*$")
        ax[1].set_ylabel("$educ^*$")
        plt.savefig("data/fwl_residual_plot.webp", format="webp", dpi=300, bbox_inches='tight')

    fwl_residual_plot()
    mo.image("data/fwl_residual_plot.webp").center()
    return (fwl_residual_plot,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Discussion 

        We have discussed the Frisch-Waugh-Lovell Theorem in-depth and have provided an intuitive approach to understanding what it means to "control" for covariates in a regression model when one is interested in a treatment parameter. It is a powerful theorem and has provided a strong underpinning for many econometric results that have developed over the years.

        FWL provides a powerful mechanism by which to visualize the relationship between an outcome and treatment after one partials out the effects from additional covariates. In fact, FWL can be used to study the relationship between any two variables and the role covariates play in explaining their underlying relationship. I recommend trying it out on a dataset where you are interested in the relationship between two variables, and the role of additional covariates in confounding that relationship!

        I hope you have gained some new knowledge from this post!

        ## References
        [1] N. Huntington-Klein, [The Effect: An Introduction to Research Design and Causality](https://medium.com/r/?url=https%3A%2F%2Ftheeffectbook.net%2F) (2022).

        [2] S. Cunningham, [Causal Inference: The Mixtape](https://medium.com/r/?url=https%3A%2F%2Fmixtape.scunning.com%2F) (2021).

        [3] M. F. Alves, [Causal Inference for the Brave and True](https://medium.com/r/?url=https%3A%2F%2Fmatheusfacure.github.io%2Fpython-causality-handbook%2Flanding-page.html) (2021).

        [4] J. Angrist & J.S. Pischke, [Mostly Harmless Econometrics: An Empiricist's Companion](https://medium.com/r/?url=https%3A%2F%2Fwww.mostlyharmlesseconometrics.com%2F) (2009). Princeton University Press.

        [5] Frisch, Ragnar, and Waugh. Partial Time Regressions as Compared with Individual Trends (1933). Econometrica: Journal of the Econometric Society, 387–401.

        [6] Lovell. Seasonal Adjustment of Economic Time Series and Multiple Regression Analysis (1963). Journal of the American Statistical Association 58 (304): 993–1010.

        [7] Lovell. A Simple Proof of the FWL Theorem (2008). Journal of Economic Education. 39 (1): 88–91.

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
