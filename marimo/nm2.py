import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    import sympy as sm

    import os

    try:
        os.chdir("assets/articles/notebooks")
    except:
        pass
    return Poly3DCollection, animation, mo, np, os, plt, sm


@app.cell
def _(mo):
    mo.md(
        r"""
        # Optimization, Newton's Method, & Profit Maximization: Part 2 - Constrained Optimization Theory
        <center> **Learn how to solve constrained optimization problems** </center>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Introduction

        > This article is the **2nd** in a 3 part series. In the <a href="/articles/nm1" target="_blank" rel="noopener noreferrer">1st part</a>, we studied basic optimization theory. Now, in pt. 2, we will extend this theory to constrained optimization problems. Lastly, in <a href="/articles/nm3" target="_blank" rel="noopener noreferrer">pt. 3</a>, we will apply the optimization theory covered, as well as econometric and economic theory, to solve a profit maximization problem

        Consider the following problem: You want to determine how much money to invest in specific financial instruments to maximize your return on investment. However, the problem of simply maximizing your return on investment is too broad and simple of an optimization question to ask. By virtue of the simplicity, the solution is to just invest all of your money in the financial instrument has the highest probability for the highest return. Clearly this is not a good investment strategy; so, how can we improve this? By putting constraints on the investment decisions, our choice variables. For example, we can specify constraints that, to name a couple, 1) limit the amount of financial risk we are willing to entertain (see [modern portfolio theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)) or 2) specify the amount of our portfolio to be allocated towards each category of financial instruments (equity, bonds, derivatives, etc.) — the possibilities are endless. Notice how this problem becomes significantly more tractable as we add constraints. Despite this simple example, it helps to capture a fundamental motivation of constrained optimization:

        > The essence of constrained optimization is to provide unconstrained optimization problems a sense of tractability and applicability to complex real world problems.


        Constrained optimization is defined as “the process of optimizing an objective function with respect to some variables in the presence of constraints on those variables.”[1] The process of adding constraints on the variables transforms an unconstrained and, perhaps, intractable optimization problem into one which can help model and solve a real world problem. However, the addition of constraints can turn a simple optimization problem into a problem that is no longer trivial. In this post, we will dive into some of the techniques that we can add to our toolbox to extend the unconstrained optimization theory, learned in part 1 of this series, to now solve constrained optimization problems.

        > In <a href="/articles/nm1" target="_blank" rel="noopener noreferrer">part 1</a>, we covered basic optimization theory — including 1) setting up and solving a simple single variable optimization problem analytically, 2) iterative optimization schemes — namely, gradient descent & Newton’s Method, and 3) implementing Newton’s method by hand and in python for a multi-dimensional optimization problem. This article is designed to be accessible for those who are already familiar with the content covered in part 1.

        ## Optimization Basics - Part 1 Recap

        A mathematical optimization problem can be formulated abstractly as such:

        $$
        \begin{equation}
        \begin{aligned}
        \min_{\mathbf{x}} \quad& f(\mathbf{x}), \mathbf{x}=[x_1,x_2,\dots,x_n]^T \in \mathbb{R}^n \\
        \text{subject to} \quad & g_j(\mathbf{x}) \le 0, j=1,2,\dots,m \\
        & h_j(\mathbf{x}) = 0, j=1,2,\dots,r 
        \end{aligned}
        \tag{1}
        \end{equation}
        $$

        where we choose real values of the vector $\mathbf{x}$ that minimize the objective function $f(\mathbf{x})$ (or maximize -$f(\mathbf{x})$) subject to the inequality constraints $g(x)$ and equality constraints $h(x)$. In part 1, we discussed how to solve these problems in the absence of $g(x)$ and $h(x)$ and now we will introduce these back into our optimization problem. First, let’s succinctly recap how to implement Newton’s method for unconstrained problems.

        Recall that we can approximate the first order necessary condition of a minimum using a Taylor Series expansion:


        $$
        \begin{equation}
        0 = \nabla f(\mathbf{x}^*)=\nabla f(\mathbf{x}_k + \Delta) = \nabla f(\mathbf{x}_k) + \mathbf{H}(\mathbf{x}_k)\Delta\Rightarrow \Delta = -\mathbf{H}^{-1}(\mathbf{x}_k)\nabla f(\mathbf{x}_k)
        \tag{2}
        \end{equation}
        $$

        where $\mathbf{H}(\mathbf{x})$ and $\nabla f(\mathbf{x})$ denote the Hessian and gradient of $f(\mathbf{x})$, respectively. Each iterative addition of delta, $\Delta$, is an expected better approximation of the optimal values $\mathbf{x}^*$. Thus, each iterative step using the NM can be represented as follows:

        $$
        \begin{equation}
        \mathbf{x}_{k+1} = \mathbf{x}_k -\mathbf{H}^{-1}(\mathbf{x}_k)\nabla f(\mathbf{x}_k)
        \tag{3}
        \end{equation}
        $$

        We do this scheme until we reach convergence across one or more of the following criteria:

        $$
        \begin{equation}
        \begin{aligned}
        &\text{Criteria 1: } \lVert \mathbf{x}_k - \mathbf{x}_{k-1} \rVert < \epsilon_1 \\[6pt]
        &\text{Criteria 2: } \lvert f(\mathbf{x}_k) - f(\mathbf{x}_{k-1}) \rvert < \epsilon_2
        \end{aligned}
        \tag{4}
        \end{equation}
        $$

        Putting this into python code, we make use of [SymPy](https://www.sympy.org/en/index.html) — a python library for symbolic mathematics — and create generalizable functions to compute the gradient, compute the Hessian, and implement Newton’s method for an n-dimensional function (see <a href="/articles/nm1" target="_blank" rel="noopener noreferrer">part 1</a> for full recap) and, leveraging these functions, we can solve an unconstrained optimization problem as follows:
        """
    )
    return


@app.cell(hide_code=True)
def _(np, sm):
    # Functions constructed in Part 1

    def get_gradient(
        function: sm.Expr,
        symbols: list[sm.Symbol],
        x0: dict[sm.Symbol, float],  # Add x0 as argument
    ) -> np.ndarray:
        """
        Calculate the gradient of a function at a given point.

        Args:
            function (sm.Expr): The function to calculate the gradient of.
            symbols (list[sm.Symbol]): The symbols representing the variables in the function.
            x0 (dict[sm.Symbol, float]): The point at which to calculate the gradient.

        Returns:
            numpy.ndarray: The gradient of the function at the given point.
        """
        d1 = {}
        gradient = np.array([])

        for i in symbols:
            d1[i] = sm.diff(function, i, 1).evalf(subs=x0)  # add evalf method
            gradient = np.append(gradient, d1[i])

        return gradient.astype(np.float64)  # Change data type to float

    def get_hessian(
        function: sm.Expr,
        symbols: list[sm.Symbol],
        x0: dict[sm.Symbol, float],
    ) -> np.ndarray:
        """
        Calculate the Hessian matrix of a function at a given point.

        Args:
        function (sm.Expr): The function for which the Hessian matrix is calculated.
        symbols (list[sm.Symbol]): The list of symbols used in the function.
        x0 (dict[sm.Symbol, float]): The point at which the Hessian matrix is evaluated.

        Returns:
        numpy.ndarray: The Hessian matrix of the function at the given point.
        """
        d2 = {}
        hessian = np.array([])

        for i in symbols:
            for j in symbols:
                d2[f"{i}{j}"] = sm.diff(function, i, j).evalf(subs=x0)
                hessian = np.append(hessian, d2[f"{i}{j}"])

        hessian = np.array(np.array_split(hessian, len(symbols)))

        return hessian.astype(np.float64)

    def newtons_method(
        function: sm.Expr,
        symbols: list[sm.Symbol],
        x0: dict[sm.Symbol, float],
        iterations: int = 100,
        tolerance: float = 10e-5,
        verbose: int = 1,
    ) -> dict[sm.Symbol, float] or None:
        """
        Perform Newton's method to find the solution to the optimization problem.

        Args:
            function (sm.Expr): The objective function to be optimized.
            symbols (list[sm.Symbol]): The symbols used in the objective function.
            x0 (dict[sm.Symbol, float]): The initial values for the symbols.
            iterations (int, optional): The maximum number of iterations. Defaults to 100.
            tolerance (float, optional): Threshold for determining convergence.
            verbose (int, optional): Control verbosity of output. 0 is no output, 1 is full output.

        Returns:
            dict[sm.Symbol, float] or None: The solution to the optimization problem, or None if no solution is found.
        """

        x_star = {}
        x_star[0] = np.array(list(x0.values()))

        if verbose != 0:
            print(f"Starting Values: {x_star[0]}")

        for i in range(iterations):
            gradient = get_gradient(function, symbols, dict(zip(x0.keys(), x_star[i])))
            hessian = get_hessian(function, symbols, dict(zip(x0.keys(), x_star[i])))

            x_star[i + 1] = x_star[i].T - np.linalg.inv(hessian) @ gradient.T

            if np.linalg.norm(x_star[i + 1] - x_star[i]) < tolerance:
                solution = dict(zip(x0.keys(), [float(x) for x in x_star[i + 1]]))
                if verbose != 0:
                    print(
                        f"\nConvergence Achieved ({i+1} iterations): Solution = {solution}"
                    )
                break
            else:
                solution = None

            if verbose != 0:
                print(f"Step {i+1}: {x_star[i+1]}")

        return solution

    return get_gradient, get_hessian, newtons_method


@app.cell
def _(newtons_method, sm):
    def unconstrained_rosenbrocks():
        x, y = sm.symbols("x y")
        Gamma = [x, y]
        objective = 100 * (y - x**2) ** 2 + (1 - x) ** 2  # Objective function
        Gamma0 = {x: -1.2, y: 1}  # Initial Guess

        return newtons_method(objective, Gamma, Gamma0)

    _ = unconstrained_rosenbrocks()
    return (unconstrained_rosenbrocks,)


@app.cell
def _(mo):
    mo.md(
        r"""
        If all of the material reviewed above feels extremely foreign, then I recommend taking a look at <a href="/articles/nm1" target="_blank" rel="noopener noreferrer">part 1</a> for a full recap. Without further ado, let’s dive into implementing constraints in our optimization problems.

        ## Solving Constrained Optimization Problems

        > Note: All of the following constrained optimization techniques can and should be incorporated w/ gradient descent algorithms when applicable!

        As we discussed above there are two possible constraints on an objective function — equality and inequality constraints. Note that there are varying methodologies out there for dealing with each type of constraint with varying pros and cons. See [2] for a further discussion of different methodologies. Nevertheless, we will hone our focus in on two methodologies, one for equality and one for inequality constraints, that I believe are robust in their performance, easy to grasp for newcomers, and easily integrated together into one cohesive problem.

        ### Equality Constraints - The Langrangian

        First, we will address optimization problems with equality constraints in our optimization problem. That is, optimization problems that take the form:

        $$
        \begin{equation}
        \begin{aligned}
        \min_{\mathbf{x}} \quad& f(\mathbf{x}), \mathbf{x}=[x_1,x_2,\dots,x_n]^T \in \mathbb{R}^n \\
        \text{subject to} \quad& h_j(\mathbf{x}) = 0, j=1,2,\dots,r 
        \end{aligned}
        \tag{5}
        \end{equation}
        $$

        Suppose we are working with the Rosenbrock’s Parabolic Valley, as in part 1, but now with the equality constraint that $x^2 - y = 2$:

        $$
        \begin{equation}
        \begin{aligned}
        \min_{\Gamma} \quad& 100(y-x^2)^2+(1-x)^2, \Gamma = \begin{bmatrix} x \\ y \end{bmatrix} \in \mathbb{R}^2 \\
        \text{subject to} \quad& x^2-y =2 \Leftrightarrow x^2-y-2=0
        \end{aligned}
        \tag{6}
        \end{equation}
        $$

        Note that, for simplicity and consistency, the equality constraints should be written such that they are equal to zero. Now our optimization problem looks like:
        """
    )
    return


@app.cell(hide_code=True)
def _(animation, mo, np, plt):
    def eqc_rosenbrocks_viz_3d():
        x = np.outer(np.linspace(-10, 10, 50), np.ones(50))
        y = x.copy().T
        z = 100 * (y - x**2) ** 2 + (1 - x) ** 2

        # Constraint
        xs = np.linspace(-np.sqrt(12), np.sqrt(12), 500)
        zs = np.linspace(-1.2e6, 1.2e6, 500)
        X, Z = np.meshgrid(xs, zs)
        Y = X**2 - 2

        # Constraint Intersection
        X2 = np.linspace(-np.sqrt(12), np.sqrt(12), 500)
        Y2 = X2**2 - 2
        Z2 = 100 * (Y2 - X2**2) ** 2 + (1 - X2) ** 2

        fig = plt.figure()
        # syntax for 3-D plotting
        ax = plt.axes(projection="3d")
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_yticks([-10, -5, 0, 5, 10])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.plot_surface(X, Y, Z, color="lightgreen", alpha=0.7, zorder=3)
        ax.plot_surface(x, y, z, cmap="plasma", alpha=0.4, zorder=2)
        ax.scatter(X2, Y2, Z2, color="green", alpha=1, zorder=3)

        # Rotating Visualization
        def rotate(angle):
            ax.view_init(azim=angle)

        rot_animation = animation.FuncAnimation(
            fig, rotate, frames=np.arange(0, 362, 2), interval=100
        )

        rot_animation.save("data/eqc_rosenbrocks_viz_3d.gif", dpi=200)

    eqc_rosenbrocks_viz_3d()
    mo.image("data/eqc_rosenbrocks_viz_3d.gif", height=500).center()
    return (eqc_rosenbrocks_viz_3d,)


@app.cell(hide_code=True)
def _(mo, np, plt):
    def eqc_rosenbrocks_viz_contour():
        # Define the Rosenbrock function
        def rosenbrock(x, y):
            return 100 * (y - x**2) ** 2 + (1 - x) ** 2

        # Compute gradient
        def grad_rosenbrock(x, y):
            df_dx = -400 * x * (y - x**2) - 2 * (1 - x)
            df_dy = 200 * (y - x**2)
            return df_dx, df_dy

        # Define the grid
        x_vals = np.linspace(-4, 4, 100)
        y_vals = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = rosenbrock(X, Y)

        # Compute gradients for quiver plot
        dX, dY = grad_rosenbrock(X, Y)

        # Define constraint: y = x^2 - 2
        x_constraint = np.linspace(-np.sqrt(6), np.sqrt(6), 500)
        y_constraint = x_constraint**2 - 2

        # Plot contours of Rosenbrock function
        plt.figure(dpi=125)
        contour = plt.contour(X, Y, Z, levels=50, cmap="plasma")
        plt.colorbar(contour)

        # Overlay gradient field
        plt.quiver(X, Y, dX, dY, color="red", alpha=0.6)

        # Mark the optimization point (theoretical minimum at (1,1))
        plt.scatter(
            1,
            1,
            color="red",
            marker="x",
            s=100,
            label="Unconstrained Optimum (1,1)",
            zorder=3,
        )

        plt.scatter(
            1,
            -1,
            color="green",
            marker="o",
            s=100,
            label="Constrained Optimum (1,-1)",
            zorder=3,
        )

        # Plot constraint curve
        plt.plot(
            x_constraint,
            y_constraint,
            color="green",
            linestyle="--",
            linewidth=1,
            label="Constraint: $y = x^2 - 2$; Feasible Region",
        )

        # Labels and legend
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Contour Representation")
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4))
        plt.savefig("data/eqc_rosenbrocks_viz_contour.webp", format="webp", dpi=200)

    eqc_rosenbrocks_viz_contour()
    mo.image("data/eqc_rosenbrocks_viz_contour.webp", height=500).center()
    return (eqc_rosenbrocks_viz_contour,)


@app.cell
def _(mo):
    mo.md(
        r"""
        where the **feasible region** of the optimal values lie along the intersection of the equality constraint curve and our objective function above.

        Joseph-Louis Lagrange developed a method for incorporating an equality constraint directly into the objective function — creating the Lagrangian function — so that traditional approaches using first and second derivates can still be applied.[2][3] Formally, the Lagrangian function takes the following form:

        $$
        \begin{equation}
        \begin{aligned}
        \mathcal{L}
        (\mathbf{x},\Lambda) &= f(\mathbf{x})+\sum^r_{j=1}\lambda_jh_j(\mathbf{x}), \\
        \mathbf{x}&=[x_1,x_2,\dots,x_n] \\
        \Lambda &= [ \lambda_1, \lambda_2, \dots,\lambda_r]
        \end{aligned}
        \tag{7}
        \end{equation}
        $$

        where $f(\mathbf{x})$ and $h(\mathbf({x})$ are the objective function and equality constraints, respectively. $\Lambda$ are the Lagrange multipliers that correspond to each equality constraint $h_j$. The Lagrange multipliers are treated as new choice variables in the Lagrangian function. It just so happens that the necessary conditions for $\mathbf{x}^*$ to be a minimum of the equality constrained problem is that $\mathbf{x}^*$ corresponds to the stationarity points of the Lagrangian $(\mathbf{x}^*, \Lambda^*)$. That is,

        $$
        \begin{equation}
        \begin{aligned}
        \frac{\partial{\mathcal{L}}}{\partial{x_i}}(\mathbf{x}^*, \Lambda^*)=0, i=1,2,\dots,n \\[8pt]
        \frac{\partial{\mathcal{L}}}{\partial{\lambda_i}}(\mathbf{x}^*, \Lambda^*)=0, i=1,2,\dots,n
        \end{aligned}
        \tag{8}
        \end{equation}
        $$

        For our above example — eq. 6 — we can write our Lagrangian function as follows:

        $$
        \begin{equation}
        \mathcal{L}(\Gamma,\lambda) = 100(y-x^2)^2+(1-x)^2+\lambda(x^2-y-2)
        \tag{9}
        \end{equation}
        $$

        We can then solve this Lagrangian using Newton’s method (or gradient descent!), but now including the Lagrange multipliers as additional choice variables.
        """
    )
    return


@app.cell
def _(newtons_method, sm):
    def eqc_rosenbrocks():
        x, y, λ = sm.symbols("x y λ")

        lagrangian = 100 * (y - x**2) ** 2 + (1 - x) ** 2 + λ * (x**2 - y - 2)
        Gamma = [x, y, λ]
        Gamma0 = {x: -1.2, y: 1, λ: 1}

        return newtons_method(lagrangian, Gamma, Gamma0)

    _ = eqc_rosenbrocks()
    return (eqc_rosenbrocks,)


@app.cell
def _(mo):
    mo.md(
        r"""
        One can easily verify that the solution satisfies our equality constraint. And there you have it! That wasn’t too bad, right? This method can be extended to add any number of equality constraints — just add another Lagrange multiplier. Let’s move on now to the incorporation of inequality constraints.

        ### Inequality Constraints - The Logarithmic Barrier Function

        Now we will address optimization problems with inequality constraints in our optimization problem. That is, optimization problems that take the form:

        $$
        \begin{equation}
        \begin{aligned}
        \min_{\mathbf{x}} \quad& f(\mathbf{x}), \mathbf{x}=[x_1,x_2,\dots,x_n]^T \in \mathbb{R}^n \\
        \text{subject to} \quad& g_j(\mathbf{x}) \le 0, j=1,2,\dots,m
        \end{aligned}
        \tag{10}
        \end{equation}
        $$


        Suppose, again, we are working with Rosenbrock’s Parabolic Valley but now with the inequality constraints $x \le 0$ and $y \ge 3$:

        $$
        \begin{equation}
        \begin{aligned}
        \min_{\Gamma} \quad& 100(y-x^2)^2+(1-x)^2, \Gamma = \begin{bmatrix} x \\ y \end{bmatrix} \in \mathbb{R}^2 \\
        \text{subject to} \quad& x \le 0, \quad y \ge 3
        \end{aligned}
        \tag{11}
        \end{equation}
        $$

        Now our optimization problem looks like:
        """
    )
    return


@app.cell(hide_code=True)
def _(Poly3DCollection, animation, mo, np, plt):
    def ineqc_rosenbrocks_viz_3d():
        # Define Rosenbrock function
        x = np.outer(np.linspace(-10, 10, 50), np.ones(50))
        y = x.copy().T
        z = 100 * (y - x**2) ** 2 + (1 - x) ** 2

        # Constraints
        # y >= 3
        xc1 = np.linspace(-10, 10, 15)
        zc1 = np.linspace(-1.2e6, 1.2e6, 15)
        XC1, ZC1 = np.meshgrid(xc1, zc1)
        YC1 = 3  # Horizontal plane

        # x <= 0
        yc2 = np.linspace(-10, 10, 20)
        zc2 = np.linspace(-1.2e6, 1.2e6, 20)
        YC2, ZC2 = np.meshgrid(yc2, zc2)
        XC2 = 0  # Vertical plane

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_yticks([-10, -5, 0, 5, 10])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Plot infeasible region boundaries with transparency
        ax.plot_surface(XC1, YC1, ZC1, color="black", alpha=0.6)
        ax.plot_surface(XC2, YC2, ZC2, color="black", alpha=0.6)

        # Plot Rosenbrock function surface
        ax.plot_surface(x, y, z, cmap="plasma", alpha=0.8)

        # Highlight feasible region
        # Feasible region vertices (clockwise order)
        feasible_region = [
            [-10, 3, float(zc2.min())],  # Bottom-left
            [0, 3, float(zc2.min())],  # Bottom-right
            [0, 10, float(zc2.min())],  # Top-right
            [-10, 10, float(zc2.min())],  # Top-left
            [-10, 3, float(zc2.max())],  # Upper-bottom-left
            [0, 3, float(zc2.max())],  # Upper-bottom-right
            [0, 10, float(zc2.max())],  # Upper-top-right
            [-10, 10, float(zc2.max())],  # Upper-top-left
        ]

        # Define faces of the feasible region box
        faces = [
            [
                feasible_region[0],
                feasible_region[1],
                feasible_region[2],
                feasible_region[3],
            ],  # Bottom face
            [
                feasible_region[4],
                feasible_region[5],
                feasible_region[6],
                feasible_region[7],
            ],  # Top face
            [
                feasible_region[0],
                feasible_region[1],
                feasible_region[5],
                feasible_region[4],
            ],  # Front face
            [
                feasible_region[2],
                feasible_region[3],
                feasible_region[7],
                feasible_region[6],
            ],  # Back face
            [
                feasible_region[1],
                feasible_region[2],
                feasible_region[6],
                feasible_region[5],
            ],  # Right face
            [
                feasible_region[0],
                feasible_region[3],
                feasible_region[7],
                feasible_region[4],
            ],  # Left face
        ]

        # Add feasible region as a green shaded box
        feasible_poly = Poly3DCollection(faces, color="lightgreen", alpha=0.4)
        ax.add_collection3d(feasible_poly)

        # Rotating Visualization
        def rotate(angle):
            ax.view_init(azim=angle)

        rot_animation = animation.FuncAnimation(
            fig, rotate, frames=np.arange(0, 362, 2), interval=100
        )

        rot_animation.save("data/ineqc_rosenbrocks_viz_3d.gif", dpi=200)

    ineqc_rosenbrocks_viz_3d()
    mo.image("data/ineqc_rosenbrocks_viz_3d.gif", height=500).center()
    return (ineqc_rosenbrocks_viz_3d,)


@app.cell(hide_code=True)
def _(mo, np, plt):
    def ineqc_rosenbrocks_viz_contour():
        # Define the Rosenbrock function
        def rosenbrock(x, y):
            return 100 * (y - x**2) ** 2 + (1 - x) ** 2

        # Compute gradient
        def grad_rosenbrock(x, y):
            df_dx = -400 * x * (y - x**2) - 2 * (1 - x)
            df_dy = 200 * (y - x**2)
            return df_dx, df_dy

        # Define the grid
        x_vals = np.linspace(-4, 4, 100)
        y_vals = np.linspace(-4, 8, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = rosenbrock(X, Y)

        # Compute gradients for quiver plot
        dX, dY = grad_rosenbrock(X, Y)

        # Plot contours of Rosenbrock function
        plt.figure(dpi=125)
        contour = plt.contour(X, Y, Z, levels=50, cmap="plasma")
        plt.colorbar(contour)

        # Overlay gradient field
        plt.quiver(X, Y, dX, dY, color="red", alpha=0.6)

        # Mark the optimization points
        plt.scatter(
            1,
            1,
            color="red",
            marker="x",
            s=100,
            label="Unconstrained Optimum (1,1)",
            zorder=3,
        )
        plt.scatter(
            -1.73,
            3,
            color="green",
            marker="o",
            s=100,
            label="Constrained Optimum (-1.73,3)",
            zorder=3,
        )

        # Plot constraint lines
        plt.axvline(
            0, color="black", linestyle="-", linewidth=2, label="Constraint: $x \leq 0$"
        )
        plt.axhline(
            3,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Constraint: $y \geq 3$",
        )

        # Shade infeasible regions in gray
        plt.fill_betweenx(
            y_vals, 0, 4, color="gray", alpha=0.5
        )  # Shade region where x > 0
        plt.fill_between(
            x_vals, -4, 3, color="gray", alpha=0.5
        )  # Shade region where y < 3

        # Labels and legend
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Contour Representation")
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4))
        plt.savefig("data/ineqc_rosenbrocks_viz_contour.webp", format="webp", dpi=200)

    ineqc_rosenbrocks_viz_contour()
    mo.image("data/ineqc_rosenbrocks_viz_contour.webp", height=500).center()
    return (ineqc_rosenbrocks_viz_contour,)


@app.cell
def _(mo):
    mo.md(
        r"""
        where the **feasible region** lies in the quadrant bounded by the constraints that is marked by the green box in the 3D plot or the unshaded region in the contour plot.

        Because these constraints do not have a strict equality, our ability to directly include them into the objective function is not as straightforward. However, we can get creative — what we can do is augment our objective function to include a “barrier” in the objective function that penalizes values of the solution that approach the bounds of the inequality constraints. These class of methods are known as “interior-point methods” or “barrier methods.”[4][5] Like the Lagrangian function, we can transform our original constrained optimization problem into an unconstrained optimization problem by incorporating barrier functions (the logarithmic barrier function in our case) that can be solved using traditional methods— thereby creating the **barrier function**. Formally, the logarithmic barrier function is characterized by:

        $$
        \begin{equation}
        \begin{aligned}
        \mathcal{B}
        (\mathbf{x},\rho) &= f(\mathbf{x})- \rho\sum^m_{j=1}\log(c_j(\mathbf{x})), \\
        \mathbf{x}&=[x_1,x_2,\dots,x_n] \\[6pt]
        c_j(\mathbf{x}) &= \begin{cases} 
        g_j(\mathbf{x}), & g_j(\mathbf{x}) \geq 0 \\
        -g_j(\mathbf{x}), & g_j(\mathbf{x}) < 0
        \end{cases}
        \end{aligned}
        \tag{12}
        \end{equation}
        $$

        where $\rho$ is a small positive scalar — known as the barrier parameter. As $\rho \rightarrow 0$, the solution of the barrier function $\mathcal{B}(\mathbf{X},\rho)$ should converge to the solution of our original constrained optimization function. Note, the $c(x)$ states that depending on how we formulate our inequality constraints (greater than or less than zero) will dictate whether we use the negative or positive of that constraint. We know that $y=\log(x)$ is undefined for $x \le 0$, thus we need to formulate our constraint to always be $\ge 0$.

        How exactly does the barrier method work, you may ask? To begin with, when using the barrier method, we must choose starting values that are in the feasible region. As the optimal values approach the “barrier” outlined by the constraint, this method relies on the fact that the logarithmic function approaches negative infinity as the value approaches zero, thereby penalizing the objective function value. As $\rho \rightarrow 0$, the penalization decreases (see figure directly below) and we converge to the solution. However, it is necessary to start with a sufficiently large $\rho$ so that the penalization is large enough to prevent “jumping” out of the barriers. Therefore, the algorithm has one extra loop than Newton’s method alone — namely, we choose a starting value $\rho$, optimize the barrier function using Newton’s method, then update $\rho$ by slowly decreasing it ($\rho \rightarrow 0$), and repeat until convergence.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    def logarithmic_barrier_function():
        # Defining surface and axes
        x = np.linspace(0.01, 20, 1000)
        y = np.log(x)
        x2 = np.linspace(0.000000000000000000001, 20, 1000)
        y2 = 0.1 * np.log(x2)

        fig = plt.figure(dpi=125)
        ax = fig.add_subplot(1, 1, 1)
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        ax.set_yticks([-4, -3, -2, -1, 1, 2, 3])
        ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

        ax.text(x=16, y=3.2, s="ρ = 1")
        ax.text(x=16, y=0.6, s="ρ = 0.1")

        # plot the function
        plt.plot(x, y, "r")
        plt.plot(x, y2, "g")

        plt.savefig("data/logarithmic_barrier_function.webp", format="webp", dpi=200)

    logarithmic_barrier_function()
    mo.image("data/logarithmic_barrier_function.webp", height=500).center()
    return (logarithmic_barrier_function,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Revisiting our example above — eq. 11 — we can write our barrier function as follows:

        $$
        \begin{equation}
        \mathcal{B}
        (\Gamma,\rho)=100(y-x^2)^2+(1-x)^2-\rho\log((y-3)(-x))
        \tag{13}
        \end{equation}
        $$

        Recall that $\log(a) + \log(b) = \log(ab)$ and our one constraint $x \le 0 \rightarrow -x \ge 0$. We must then update our code to accommodate the barrier method algorithm:
        """
    )
    return


@app.cell
def _(newtons_method, np, sm):
    def constrained_newtons_method(
        function: sm.Expr,
        symbols: list[sm.Symbol],
        x0: dict[sm.Symbol, float],
        rho_steps: int = 100,
        discount_rate: float = 0.9,
        newton_method_iterations: int = 100,
        tolerance: float = 10e-5,
    ) -> dict[sm.Symbol, float] | None:
        """
        Performs constrained Newton's method to find the optimal solution of a function subject to constraints.

        Args:
            function (sm.Expr): The function to optimize.
            symbols (list[sm.Symbol]): The symbols used in the function.
            x0 (dict[sm.Symbol, float]): The initial values for the symbols.
            rho_steps (int, optional): The number of steps to update rho. Defaults to 100.
            discount_rate (float, optional): The scalar to discount rho by at each step. Default is 0.9.
            newton_method_iterations (int, optional): The maximum number of iterations in Newton Method internal loop. Defaults to 100.
            tolerance (float, optional): Threshold for determining convergence.

        Returns:
            dict[sm.Symbol, float] or None: The optimal solution if convergence is achieved, otherwise None.
        """

        rho = list(x0.keys())[-1]
        optimal_solutions = []
        optimal_solutions.append(x0)

        for step in range(rho_steps):
            if step % 10 == 0:
                print("\n" + "===" * 20)
                print(f"Step {step} w/ rho={optimal_solutions[step][rho]}")
                print("===" * 20 + "\n")
                print(f"Current solution: {optimal_solutions[step]}")

            function_eval = function.evalf(subs={rho: optimal_solutions[step][rho]})

            values = optimal_solutions[step].copy()
            del values[rho]

            optimal_solution = newtons_method(
                function_eval,
                symbols[:-1],
                values,
                iterations=newton_method_iterations,
                tolerance=tolerance,
                verbose=0,
            )

            optimal_solutions.append(optimal_solution)

            # Check for overall convergence
            current_solution = np.array(
                [v for k, v in optimal_solutions[step].items() if k != rho]
            )
            previous_solution = np.array(
                [v for k, v in optimal_solutions[step - 1].items() if k != rho]
            )
            if np.linalg.norm(current_solution - previous_solution) < tolerance:
                overall_solution = optimal_solutions[step]
                del overall_solution[rho]
                print(
                    f"\n Overall Convergence Achieved ({step} steps): Solution = {overall_solution}\n"
                )
                break
            else:
                overall_solution = None

            # Update rho
            optimal_solutions[step + 1][rho] = (
                discount_rate * optimal_solutions[step][rho]
            )

        return overall_solution

    return (constrained_newtons_method,)


@app.cell
def _(mo):
    mo.md(
        r"""We can now solve the Barrier function with the code above (Note: Make sure starting values are in the feasible range of inequality constraints & you may have to increase the starting value of rho if you jump out of inequality constraints):"""
    )
    return


@app.cell
def _(constrained_newtons_method, sm):
    def ineqc_rosenbrocks():
        x, y, ρ = sm.symbols("x y ρ")

        Barrier_objective = (
            100 * (y - x**2) ** 2 + (1 - x) ** 2 - ρ * sm.log((-x) * (y - 3))
        )
        Gamma = [x, y, ρ]  # Function requires last symbol to be ρ!
        Gamma0 = {x: -15, y: 15, ρ: 10}

        return constrained_newtons_method(Barrier_objective, Gamma, Gamma0)

    _ = ineqc_rosenbrocks()
    return (ineqc_rosenbrocks,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Again, one can verify the solution satisfies the inequality constraints specified. And there you have it. We have now tackled inequality constraints in our optimization problems. To wrap up, let’s put everything together and move on to tackling constrained optimization problems with mixed constraints — which is simply the combination of what we have done above.

        ### Putting it All Together

        Let’s now solve our optimization problem by combining both the equality and inequality constraints from above. That is, we want to solve an optimization of the form:

        $$
        \begin{equation}
        \begin{aligned}
        \min_{\mathbf{x}} \quad& f(\mathbf{x}), \mathbf{x}=[x_1,x_2,\dots,x_n]^T \in \mathbb{R}^n \\
        \text{subject to} \quad & g_j(\mathbf{x}) \le 0, j=1,2,\dots,m \\
        & h_j(\mathbf{x}) = 0, j=1,2,\dots,r 
        \end{aligned}
        \tag{14}
        \end{equation}
        $$

        All we have to do is combine the Lagrangian and the Barrier functions into one function. Thus, we can create a generalizable function, call it O, for dealing with optimization problems that have both equality and inequality constraints:

        $$
        \begin{equation}
        \begin{aligned}
        \mathcal{O}
        (\mathbf{x},\Lambda,\rho) &= f(\mathbf{x}) + \sum^r_{j=1}\lambda_jh_j(\mathbf{x})- \rho\sum^m_{j=1}\log(c_j(\mathbf{x})), \\
        \mathbf{x}&=[x_1,x_2,\dots,x_n] \\[6pt]
        \Lambda &= [\lambda_1,\lambda_2,\dots,\lambda_r] \\[6pt]
        c_j(\mathbf{x}) &= \begin{cases} 
        g_j(\mathbf{x}), & g_j(\mathbf{x}) \geq 0 \\
        -g_j(\mathbf{x}), & g_j(\mathbf{x}) < 0
        \end{cases}
        \end{aligned}
        \tag{15}
        \end{equation}
        $$

        where, as before, $\Lambda$ is the vector of Lagrange multipliers and $\rho$ is the barrier parameter. Thus, combining our constrained (Eq. 6) and unconstrained problems from above (Eq. 11), we can formulate our mixed constrained optimization problem as follows:

        $$
        \begin{equation}
        \begin{aligned}
        \mathcal{O}
        (\Gamma,\Lambda,\rho) &= 100(y-x^2)^2+(1-x)^2+\lambda(x^2-y-2)-\rho \times \log((y-3)(-x))
        \end{aligned}
        \tag{16}
        \end{equation}
        $$

        In python,
        """
    )
    return


@app.cell
def _(constrained_newtons_method, sm):
    def combined_rosenbrocks():
        x, y, λ, ρ = sm.symbols("x y λ ρ")

        combined_objective = (
            100 * (y - x**2) ** 2
            + (1 - x) ** 2
            + λ * (x**2 - y - 2)
            - ρ * sm.log((-x) * (y - 3))
        )
        Gamma = [x, y, λ, ρ]  # Function requires last symbol to be ρ!
        Gamma0 = {x: -15, y: 15, λ: 0, ρ: 10}

        return constrained_newtons_method(combined_objective, Gamma, Gamma0)

    _ = combined_rosenbrocks()
    return (combined_rosenbrocks,)


@app.cell
def _(mo):
    mo.md(
        r"""
        And we can again verify this solution satisfies our contraints!

        ## Conclusion

        Phew. Take a deep breath — you earned it. Hopefully at this point you should have a much better understanding of the techniques to incorporate constraints into your optimization problems. We are still just brushing the surface of the different tools and techniques utilized in mathematical optimization.

        Stay tuned for part 3 of this series, the final part, where we will apply the optimization material learned thus far alongside econometric & economic theory to solve a profit maximization problem. It is my goal that part 3 will bring home everything we have covered and show a practical use case. As usual, I hope you have enjoyed reading this much as much I have enjoyed writing it!

        ## References

        [1] https://en.wikipedia.org/wiki/Constrained_optimization

        [2] Snyman, J. A., & Wilke, D. N. (2019). Practical mathematical optimization: Basic optimization theory and gradient-based algorithms (2nd ed.). Springer.

        [3] https://en.wikipedia.org/wiki/Lagrange_multiplier

        [4] https://en.wikipedia.org/wiki/Interior-point_method

        [5] https://en.wikipedia.org/wiki/Barrier_function

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
