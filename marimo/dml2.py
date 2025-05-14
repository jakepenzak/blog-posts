import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # Relevant Imports

    import marimo as mo
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import graphviz
    import matplotlib.pyplot as plt
    from IPython.display import display, HTML

    import os

    try:
        os.chdir("assets/articles/notebooks")
    except:
        pass

    np.random.seed(00)

    ## Helper Plots

    COLORS = ["#00B0F0", "#FF0000", "#B0F000"]

    def plot_effect(
        effect_true, effect_pred, save_path, figsize=(8, 5), ylim=(-10, 100)
    ):
        plt.figure(figsize=figsize)
        plt.scatter(effect_true, effect_pred, color=COLORS[0], s=10)
        plt.plot(
            np.sort(effect_true),
            np.sort(effect_true),
            color=COLORS[1],
            alpha=0.7,
            label="Perfect model",
        )
        plt.xlabel("True effect", fontsize=14)
        plt.ylabel("Predicted effect", fontsize=14)
        plt.legend()
        plt.savefig(save_path, format="webp", dpi=300, bbox_inches='tight')

    def hist_effect(effect_true, effect_pred, save_path, figsize=(8, 5)):
        plt.figure(figsize=figsize)

        plt.hist(
            effect_pred,
            color="r",
            alpha=0.8,
            density=True,
            bins=50,
            label="Linear DML CATE Prediction",
        )
        plt.hist(
            effect_true,
            color="b",
            alpha=0.4,
            density=True,
            bins=50,
            label="True CATE",
        )

        plt.legend()
        plt.savefig(save_path, format="webp", dpi=300, bbox_inches='tight')
    return (
        COLORS,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HTML,
        cross_val_predict,
        display,
        graphviz,
        hist_effect,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        os,
        pd,
        plot_effect,
        plt,
        r2_score,
        smf,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Double Machine Learning, Simplified: Part 2 - Targeting & the CATE 
        <center> **Learn how to utilize DML for estimating idiosyncratic treatment effects to enable personalized targeting** </center>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Introduction

        > This article is the **2nd** in a 2 part series on simplifying and democratizing Double Machine Learning. In the <a href="/articles/dml1" target="_blank" rel="noopener noreferrer">1st part</a>, we covered the fundamentals of Double Machine Learning, along with two basic causal inference applications. Now, in pt. 2, we will extend this knowledge to turn our causal inference problem into a prediction task, wherein we predict individual level treatment effects to aid in decision making and data-driven targeting

        Double Machine Learning, as we learned in [part 1](/articles/dml1) of this series, is a highly flexible partially-linear causal inference method for estimating the average treatment effect (ATE) of a treatment. Specifically, it can be utilized to model highly non-linear confounding relationships in observational data (especially when our set of controls/confounders is of extremely high dimensionality) and/or to reduce the variation in our key outcome in experimental settings. Estimating the ATE is particularly useful in understanding the average impact of a specific treatment, which can be extremely useful for future decision making. However, extrapolating this treatment effect assumes a degree homogeneity in the effect; that is, regardless of the population we roll treatment out to, we anticipate the effect to be similar to the ATE. What if we are limited in the number of individuals who we can target for future rollout and thus want to understand among which subpopulations the treatment was most effective to drive highly effective rollout?

        This issue described above concerns estimating treatment effect heterogeneity. That is, how does our treatment effect impact different subsets of the population? Luckily for us, DML provides a powerful framework to do exactly this. Specifically, we can make use of DML to estimate the Conditional Average Treatment Effect (CATE). First, let’s revisit our definition of the ATE, in binary and continuous cases, respectively:

        $$
        \begin{equation}
        \text{ATE}=\mathbb{E_n}[y(T=1)-y(T=0)]
        \tag{1}
        \end{equation}
        $$

        $$
        \begin{equation}
        \text{ATE}=\mathbb{E_n}\left[\frac{\partial y}{\partial T}\right]
        \tag{2}
        \end{equation}
        $$

        Now with the CATE, we estimate the ATE conditional on a set of values for our covariates, $\mathbf{X}$:

        $$
        \begin{equation}
        \text{CATE}=\mathbb{E_n}[y(T=1)-y(T=0)|\mathbf{X}=x] 
        \tag{3}
        \end{equation}
        $$

        $$
        \begin{equation}
        \text{CATE}=\mathbb{E_n}\left[\frac{\partial y}{\partial T}\right|\mathbf{X}=x]
        \tag{4}
        \end{equation}
        $$

        For example, if we wanted to know the treatment effect for males versus females, we can estimate the CATE conditional on the covariate being equal to each subgroup of interest. Note that we can estimate highly aggregated CATEs (i.e., at a male vs. female level), also known as Group Average Treatment Effects (GATEs), or we can allow $\mathbf{X}$ to take on an extremely high dimensionality and thus closely estimate each individuals treatment effect. You may immediately notice the benefits in being able to do this: we can utilize this information to make highly informed decisions in future targeting of the treatment! Even more notable, we can create a CATE function to make predictions of the treatment effect on previously unexposed individuals!

        Note, that there are many models that exist for estimating CATEs, which we'll cover in a subsequent post. For now, we'll cover two techniques within the partially linear DML formulation for estimating this CATE function; namely, Linear DML and Non-Parametric DML. Er will show how to estimate the CATE mathematically and then provide examples for each case.

        > Note: Unbiased estimation of the CATE still requires the exogeneity/CIA/Ignorability assumption to hold as covered in part 1.

        **Everything demonstrated below can and should be extended to the experimental setting (RCT or A/B Testing), where exogeneity is satisfied by construction, as covered in application 2 of part 1.**

        ## Linear DML for Estimating the CATE

        Estimating the CATE in the linear DML framework is a simple extension of DML for estimating the ATE:

        $$
        \begin{equation}
        y-\mathcal{M}_y(\mathbf{X})=\beta_0+\beta_1(T-\mathcal{M}_T(\mathbf{X}))+\epsilon 
        \tag{5}
        \end{equation}
        $$

        where $y$ is our outcome, $T$ is our treatment, & $\mathcal{M}_y$ and $\mathcal{M}_T$ are both flexible ML models (our nuisance functions) to predict $y$ and $T$ given confounders and/or controls, $\mathbf{X}$, respectively. To estimate the CATE function using Linear DML, we can simply include interaction terms of the treatment residuals with our covariates. Observe:

        $$
        \begin{equation}
        y-\mathcal{M}_y(\mathbf{X})=\beta_0+\beta_1(T-\mathcal{M}_T(\mathbf{X}))+(T-\mathcal{M}_T(\mathbf{X}))\mathbf{X}\mathbf{\Omega} + \epsilon 
        \tag{6}
        \end{equation}
        $$

        where $\mathbf{\Omega}$ is the vector of coefficients for the interaction terms. Now our CATE function, call it $\tau$, takes the form $\tau(\mathbf{X}) = \beta_1 + \mathbf{X}\mathbf{\Omega}$, where we can predict each individuals CATE given $\mathbf{X}$. If $T$ is continuous, this CATE function is for a 1 unit increase in T. Note that $\tau(\mathbf{X}) = \beta_1$ in eq. (3) where $\tau(\mathbf{X})$ is assumed a constant. Let’s take a look at this in action!

        First, let’s use the same casual DAG from part 1, where we will be looking at the effect of an individuals time spent on the website on their purchase amount, or sales, in the past month (assuming we observe all confounders).:
        """
    )
    return


@app.cell(hide_code=True)
def _(graphviz, mo):
    def create_dag():
        # Create a directed graph
        g = graphviz.Digraph(format="png")

        # Add nodes
        nodes = [
            "Age",
            "# Social Media Accounts",
            "Yrs Member",
            "Time on Website",
            "Sales",
            "Z",
        ]
        [g.node(n) for n in nodes]

        g.edge("Age", "Time on Website")
        g.edge("# Social Media Accounts", "Time on Website")
        g.edge("Yrs Member", "Time on Website")
        g.edge("Age", "Sales")
        g.edge("# Social Media Accounts", "Sales")
        g.edge("Yrs Member", "Sales")
        g.edge("Time on Website", "Sales", color="red")
        g.edge("Z", "Sales")

        g.graph_attr["dpi"] = "400"

        # Render for print
        g.render("data/dag1", format="webp")

    create_dag()
    mo.image("data/dag1.webp").center()
    return (create_dag,)


@app.cell
def _(mo):
    mo.md(r"""Let’s then simulate this DGP using a similar process as utilized in part 1 (note that all values & data are chosen and generated arbitrarily for demonstrative purposes). Observe that we now include interaction terms in the sales DGP to model the CATE, or treatment effect heterogeneity (note that the DGP in part 1 had no treatment effect heterogeneity by construction)""")
    return


@app.cell
def _(np, pd):
    # Sample Size
    N = 100_000

    # Confounders (X)
    age = np.random.randint(low=18, high=75, size=N)
    num_social_media_profiles = np.random.choice(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=N
    )
    yr_membership = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=N)

    # Arbitrary Covariates (Z)
    Z = np.random.normal(loc=50, scale=25, size=N)

    # Error Terms
    ε1 = np.random.normal(loc=20, scale=5, size=N)
    ε2 = np.random.normal(loc=40, scale=15, size=N)

    # Treatment (T = g(X) + ε1)
    def T(age, num_social_media_profiles, yr_membership, ε1):
        time_on_website = np.maximum(
            10
            - 0.01 * age
            - 0.001 * age**2
            + num_social_media_profiles
            - 0.01 * num_social_media_profiles**2
            - 0.01 * (age * num_social_media_profiles)
            + 0.2 * yr_membership
            + 0.001 * yr_membership**2
            - 0.01 * (age * yr_membership)
            + 0.2 * (num_social_media_profiles * yr_membership)
            + 0.01
            * (num_social_media_profiles * np.log(age) * age * yr_membership ** (1 / 2))
            + ε1,
            0,
        )
        return time_on_website

    time_on_website = T(age, num_social_media_profiles, yr_membership, ε1)

    # Outcome (y = f(T,X,Z) + ε2)
    def y(time_on_website, age, num_social_media_profiles, yr_membership, Z, ε2):
        sales = np.maximum(
            25
            + 5 * time_on_website  # Baseline Treatment Effect
            - 0.2 * time_on_website * age  # Heterogeneity
            + 2 * time_on_website * num_social_media_profiles  # Heterogeneity
            + 2 * time_on_website * yr_membership  # Heterogeneity
            - 0.1 * age
            - 0.001 * age**2
            + 8 * num_social_media_profiles
            - 0.1 * num_social_media_profiles**2
            - 0.01 * (age * num_social_media_profiles)
            + 2 * yr_membership
            + 0.1 * yr_membership**2
            - 0.01 * (age * yr_membership)
            + 3 * (num_social_media_profiles * yr_membership)
            + 0.1
            * (num_social_media_profiles * np.log(age) * age * yr_membership ** (1 / 2))
            + 0.5 * Z
            + ε2,
            0,
        )
        return sales

    sales = y(time_on_website, age, num_social_media_profiles, yr_membership, Z, ε2)

    df = pd.DataFrame(
        np.array(
            [sales, time_on_website, age, num_social_media_profiles, yr_membership, Z]
        ).T,
        columns=[
            "sales",
            "time_on_website",
            "age",
            "num_social_media_profiles",
            "yr_membership",
            "Z",
        ],
    )
    return (
        N,
        T,
        Z,
        age,
        df,
        num_social_media_profiles,
        sales,
        time_on_website,
        y,
        yr_membership,
        ε1,
        ε2,
    )


@app.cell(hide_code=True)
def _(df):
    df
    return


@app.cell
def _(mo):
    mo.md(r"""Now, to estimate our CATE function, as outlined in eq. (4), we can run:""")
    return


@app.cell
def _(GradientBoostingRegressor, cross_val_predict, df, smf):
    # DML Procedure for Estimating the CATE
    M_sales = GradientBoostingRegressor()
    M_time_on_website = GradientBoostingRegressor()

    df["residualized_sales"] = df["sales"] - cross_val_predict(
        M_sales,
        df[["age", "num_social_media_profiles", "yr_membership"]],
        df["sales"],
        cv=3,
    )

    df["residualized_time_on_website"] = df["time_on_website"] - cross_val_predict(
        M_time_on_website,
        df[["age", "num_social_media_profiles", "yr_membership"]],
        df["time_on_website"],
        cv=3,
    )

    DML_model = smf.ols(
        formula="residualized_sales ~ 1 + residualized_time_on_website + residualized_time_on_website:age + residualized_time_on_website:num_social_media_profiles + residualized_time_on_website:yr_membership",
        data=df,
    ).fit()

    print(DML_model.summary())
    return DML_model, M_sales, M_time_on_website


@app.cell
def _(mo):
    mo.md(r"""Here we can see that linear DML closely modeled the true DGP for the CATE (see coefficients on interaction terms in sales DGP). Let’s evaluate the performance of our CATE function by comparing the linear DML predictions to the true CATE for a 1 hour increase in time on the spent on the website:""")
    return


@app.cell
def _(
    DML_model,
    Z,
    age,
    df,
    mean_absolute_error,
    mean_squared_error,
    num_social_media_profiles,
    r2_score,
    time_on_website,
    y,
    yr_membership,
    ε2,
):
    # Predict CATE of 1 hour increase
    linear_dml_cates = DML_model.predict(
        df.assign(
            residualized_time_on_website=lambda x: x.residualized_time_on_website + 1
        )
    ) - DML_model.predict(df)

    # True CATE of 1 hour increase
    X = [age, num_social_media_profiles, yr_membership, Z, ε2]
    true_cates = y(time_on_website + 1, *X) - y(time_on_website, *X)

    print(f"Mean Squared Error: {mean_squared_error(true_cates, linear_dml_cates)}")
    print(f"Mean Absolute Error: {mean_absolute_error(true_cates, linear_dml_cates)}")
    print(f"R-Squared: {r2_score(true_cates, linear_dml_cates)}")
    return X, linear_dml_cates, true_cates


@app.cell
def _(mo):
    mo.md(r"""Plotting the distributions of the predicted CATE and true CATE, we obtain:""")
    return


@app.cell(hide_code=True)
def _(hist_effect, linear_dml_cates, mo, true_cates):
    hist_effect(true_cates, linear_dml_cates, save_path="data/linear_dml_hist.webp")

    mo.image("data/linear_dml_hist.webp", height=500).center()
    return


@app.cell
def _(mo):
    mo.md(r"""Additionally, plotting the predicted values versus the true values we obtain:""")
    return


@app.cell(hide_code=True)
def _(linear_dml_cates, mo, plot_effect, true_cates):
    plot_effect(true_cates, linear_dml_cates, save_path="data/linear_dml_line.webp")

    mo.image("data/linear_dml_line.webp", height=500).center()
    return


@app.cell
def _(mo):
    mo.md(r"""Overall, we have pretty impressive performance! However, the primary limitation in this approach is that we must manually specify the functional form of the CATE function, thus if we are only including linear interaction terms we may not capture the true CATE function. In our example, we simulated the DGP to only have these linear interaction terms and thus the performance is strong by construction, but let’s see what happens when we tweak the DGP for the CATE to be arbitrarily non-linear:""")
    return


@app.cell
def _(
    Z,
    age,
    np,
    num_social_media_profiles,
    pd,
    time_on_website,
    yr_membership,
    ε2,
):
    # Outcome (y = f(T,X,Z) + ε2)
    def y_fn_nonlinear(
        time_on_website, age, num_social_media_profiles, yr_membership, Z, ε2
    ):
        sales = np.maximum(
            25
            + 5 * time_on_website  # Baseline Treatment Effect
            - 0.2 * time_on_website * age  # Heterogeneity
            - 0.0005 * time_on_website * age**2  # Heterogeneity
            + 0.8 * time_on_website * num_social_media_profiles  # Heterogeneity
            + 0.001 * time_on_website * num_social_media_profiles**2  # Heterogeneity
            + 0.8 * time_on_website * yr_membership  # Heterogeneity
            + 0.001 * time_on_website * yr_membership**2  # Heterogeneity
            + 0.005
            * time_on_website
            * yr_membership
            * num_social_media_profiles
            * age  # Heterogeneity
            + 0.005
            * time_on_website
            * (yr_membership**3 / (1 + num_social_media_profiles**2))
            * np.log(age) ** 2
            - 0.1 * age
            - 0.001 * age**2
            + 8 * num_social_media_profiles
            - 0.1 * num_social_media_profiles**2
            - 0.01 * (age * num_social_media_profiles)
            + 2 * yr_membership
            + 0.1 * yr_membership**2
            - 0.01 * (age * yr_membership)
            + 3 * (num_social_media_profiles * yr_membership)
            + 0.1
            * (num_social_media_profiles * np.log(age) * age * yr_membership ** (1 / 2))
            + 0.5 * Z
            + ε2,
            0,
        )
        return sales

    sales_nonlinear = y_fn_nonlinear(
        time_on_website, age, num_social_media_profiles, yr_membership, Z, ε2
    )

    df_nonlinear = pd.DataFrame(
        np.array(
            [
                sales_nonlinear,
                time_on_website,
                age,
                num_social_media_profiles,
                yr_membership,
                Z,
            ]
        ).T,
        columns=[
            "sales",
            "time_on_website",
            "age",
            "num_social_media_profiles",
            "yr_membership",
            "Z",
        ],
    )
    return df_nonlinear, sales_nonlinear, y_fn_nonlinear


@app.cell
def _(mo):
    mo.md(r"""Fitting our models:""")
    return


@app.cell
def _(GradientBoostingRegressor, cross_val_predict, df_nonlinear, smf):
    # DML Procedure
    M_sales2 = GradientBoostingRegressor()
    M_time_on_website2 = GradientBoostingRegressor()

    df_nonlinear["residualized_sales"] = df_nonlinear["sales"] - cross_val_predict(
        M_sales2,
        df_nonlinear[["age", "num_social_media_profiles", "yr_membership"]],
        df_nonlinear["sales"],
        cv=3,
    )

    df_nonlinear["residualized_time_on_website"] = df_nonlinear[
        "time_on_website"
    ] - cross_val_predict(
        M_time_on_website2,
        df_nonlinear[["age", "num_social_media_profiles", "yr_membership"]],
        df_nonlinear["time_on_website"],
        cv=3,
    )

    DML_model_nonlinear = smf.ols(
        formula="residualized_sales ~ 1 + residualized_time_on_website + residualized_time_on_website:age + residualized_time_on_website:num_social_media_profiles + residualized_time_on_website:yr_membership",
        data=df_nonlinear,
    ).fit()

    print(DML_model_nonlinear.summary())
    return DML_model_nonlinear, M_sales2, M_time_on_website2


@app.cell
def _(mo):
    mo.md(r"""And then evaluating performance:""")
    return


@app.cell
def _(
    DML_model_nonlinear,
    X,
    df_nonlinear,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    time_on_website,
    y_fn_nonlinear,
):
    # Predict CATE of 1 hour increase
    linear_dml_cates_nonlinear = DML_model_nonlinear.predict(
        df_nonlinear.assign(
            residualized_time_on_website=lambda x: x.residualized_time_on_website + 1
        )
    ) - DML_model_nonlinear.predict(df_nonlinear)

    # True CATE of 1 hour increase
    true_cates_nonlinear = y_fn_nonlinear(time_on_website + 1, *X) - y_fn_nonlinear(
        time_on_website, *X
    )

    print(
        f"Mean Squared Error: {mean_squared_error(true_cates_nonlinear, linear_dml_cates_nonlinear)}"
    )
    print(
        f"Mean Absolute Error: {mean_absolute_error(true_cates_nonlinear, linear_dml_cates_nonlinear)}"
    )
    print(f"R-Squared: {r2_score(true_cates_nonlinear, linear_dml_cates_nonlinear)}")
    return linear_dml_cates_nonlinear, true_cates_nonlinear


@app.cell(hide_code=True)
def _(hist_effect, linear_dml_cates_nonlinear, mo, true_cates_nonlinear):
    hist_effect(
        true_cates_nonlinear,
        linear_dml_cates_nonlinear,
        save_path="data/linear_dml_nonlinear_hist.webp",
    )

    mo.image("data/linear_dml_nonlinear_hist.webp", height=500).center()
    return


@app.cell(hide_code=True)
def _(linear_dml_cates_nonlinear, mo, plot_effect, true_cates_nonlinear):
    plot_effect(
        true_cates_nonlinear,
        linear_dml_cates_nonlinear,
        save_path="data/linear_dml_nonlinear_line.webp",
    )

    mo.image("data/linear_dml_nonlinear_line.webp", height=500).center()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Here we see much degradation in performance. This non-linearity in the CATE function is precisely where Non-Parametric DML can shine!

        ## Non-Parametric DML for Estimating the CATE

        Non-Parametric DML goes one step further and allows for another flexible non-parametric ML model to be utilized for learning the CATE function! Let’s take a look at how we can, mathematically, do exactly this. Let $\tau(\mathbf{X})$ continue to denote our CATE function. Let’s start with defining our error term relative to eq. 3 (note we drop the intercept $\beta_0$ as this parameter is partialled out in residualization step; we could similarly drop this in the linear DML formulation, but for the sake of simplicity and consistency with part 1, we do not do this):


        $$
        \begin{align*}
        y-\mathcal{M}_y(\mathbf{X})&=\tau(\mathbf{X})(T-\mathcal{M}_T(\mathbf{X}))+\epsilon \\
        \tilde{y} &=\tau(\mathbf{X})\tilde{T}+\epsilon \\
        \epsilon&=\tilde{y}-\tau(\mathbf{X})\tilde{T}
        \end{align*}
        $$

        Then define the causal loss function as such (note this is just the MSE!):

        $$
        \begin{align*}
        \mathscr{L}(\tau(\mathbf{X})) 
        &= \frac{1}{N}\sum_{i=1}^N\bigl(\tilde{y}_i - \tau(\mathbf{X}_i)\tilde{T}_i\bigr)^2 \\
        &= \frac{1}{N}\sum_{i=1}^N\tilde{T}_i^2\bigl(\frac{\tilde{y}_i}{\tilde{T}_i} - \tau(\mathbf{X}_i)\bigr)^2
        \end{align*}
        $$

        What does this mean? We can directly learn $\tau(\mathbf{X})$ with any flexible ML model via minimizing our causal loss function! This amounts to a weighted regression problem with our target and weights, respectively, as:

        $$
        \begin{align*}
        \text{Target}&=\frac{\tilde{y}_i}{\tilde{T}_i} \\
        \text{Weights}&=\tilde{T}_i^2 \\
        \end{align*}
        $$

        _Take a moment and soak in the elegance of this result… We can directly learn the CATE function & predict an individuals CATE given our residualized outcome, $y$, and treatment, $T$!_

        Let’s take a look at this in action now. We will reuse the DGP for the non-linear CATE function that was utilized in the example where linear DML performs poorly above. To construct of Non-Parametric DML model, we can run:
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Then define the causal loss function as such (note this is just the MSE!):""")
    return


@app.cell
def _(GradientBoostingRegressor, df_nonlinear):
    # Define Target & Weights
    df_nonlinear["target"] = (
        df_nonlinear["residualized_sales"]
        / df_nonlinear["residualized_time_on_website"]
    )
    df_nonlinear["weights"] = df_nonlinear["residualized_time_on_website"] ** 2

    # Non-Parametric CATE Model
    CATE_model = GradientBoostingRegressor()
    CATE_model.fit(
        df_nonlinear[["age", "num_social_media_profiles", "yr_membership"]],
        df_nonlinear["target"],
        sample_weight=df_nonlinear["weights"],
    )
    return (CATE_model,)


@app.cell
def _(mo):
    mo.md(r"""And to make predictions + evaluate performance:""")
    return


@app.cell
def _(
    CATE_model,
    df_nonlinear,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    true_cates_nonlinear,
):
    # Predict CATE of 1 hour increase
    nonparam_dml_cates_nonlinear = CATE_model.predict(
        df_nonlinear[["age", "num_social_media_profiles", "yr_membership"]]
    )

    print(
        f"Mean Squared Error: {mean_squared_error(true_cates_nonlinear, nonparam_dml_cates_nonlinear)}"
    )
    print(
        f"Mean Absolute Error: {mean_absolute_error(true_cates_nonlinear, nonparam_dml_cates_nonlinear)}"
    )
    print(f"R-Squared: {r2_score(true_cates_nonlinear, nonparam_dml_cates_nonlinear)}")
    return (nonparam_dml_cates_nonlinear,)


@app.cell(hide_code=True)
def _(hist_effect, mo, nonparam_dml_cates_nonlinear, true_cates_nonlinear):
    hist_effect(
        true_cates_nonlinear,
        nonparam_dml_cates_nonlinear,
        save_path="data/nonparam_dml_nonlinear_hist.webp",
    )

    mo.image("data/nonparam_dml_nonlinear_hist.webp", height=500).center()
    return


@app.cell(hide_code=True)
def _(mo, nonparam_dml_cates_nonlinear, plot_effect, true_cates_nonlinear):
    plot_effect(
        true_cates_nonlinear,
        nonparam_dml_cates_nonlinear,
        save_path="data/nonparam_dml_nonlinear_line.webp",
    )

    mo.image("data/nonparam_dml_nonlinear_line.webp", height=500).center()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Here we can see that, although not perfect, the non-parametric DML approach was able to model the non-linearities in the CATE function much better than the linear DML approach. We can of course further improve the performance via tuning our model. Note that we can use explainable AI tools, such as [SHAP values](https://shap.readthedocs.io/en/latest/index.html), to understand the nature of our treatment effect heterogeneity!

        ## Conclusion

        And there you have it! Thank you for taking the time to read through my article. I hope this article has taught you how to go beyond estimating only the ATE & utilize DML to estimate the CATE to further understanding heterogeneity in the treatment effects and drive more causal inference- & data- driven targeting schemes.

        As always, I hope you have enjoyed reading this as much as I enjoyed writing it!

        ## References
        [1] V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, and a. W. Newey. Double Machine Learning for Treatment and Causal Parameters. ArXiv e-prints, July 2016.

        <div style="text-align: center; font-size: 24px;">❖❖❖</div>

        <center>
        Access all the code via this Marimo Notebook or my [GitHub Repo](https://github.com/jakepenzak/blog-posts)

        I appreciate you reading my post! My posts primarily explore real-world and theoretical applications of econometric and statistical/machine learning techniques, but also whatever I am currently interested in or learning 😁. At the end of the day, I write to learn! I hope to make complex topics slightly more accessible to all.
        </center>
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
