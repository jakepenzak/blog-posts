import numpy as np
import sympy as sm


def gradient_descent(
    function: sm.core.expr.Expr,
    symbols: list[sm.core.symbol.Symbol],
    x0: dict[sm.core.symbol.Symbol, float],
    learning_rate: float = 0.1,
    iterations: int = 100,
) -> dict[sm.core.symbol.Symbol, float] or None:
    """
    Performs gradient descent optimization to find the minimum of a given function.

    Args:
        function (sm.core.expr.Expr): The function to be optimized.
        symbols (list[sm.core.symbol.Symbol]): The symbols used in the function.
        x0 (dict[sm.core.symbol.Symbol, float]): The initial values for the symbols.
        learning_rate (float, optional): The learning rate for the optimization. Defaults to 0.1.
        iterations (int, optional): The maximum number of iterations. Defaults to 100.

    Returns:
        dict[sm.core.symbol.Symbol, float] or None: The solution found by the optimization, or None if no solution is found.
    """
    x_star = {}
    x_star[0] = np.array(list(x0.values()))

    # x = [] ## Return x for visual!

    print(f"Starting Values: {x_star[0]}")

    for i in range(iterations):
        # x.append(dict(zip(x0.keys(),x_star[i]))) ## Return x for visual!

        gradient = get_gradient(function, symbols, dict(zip(x0.keys(), x_star[i])))

        x_star[i + 1] = x_star[i].T - learning_rate * gradient.T

        if np.linalg.norm(x_star[i + 1] - x_star[i]) < 10e-5:
            solution = dict(zip(x0.keys(), x_star[i + 1]))
            print(f"\nConvergence Achieved ({i+1} iterations): Solution = {solution}")
            break
        else:
            solution = None

        print(f"Step {i+1}: {x_star[i+1]}")

    return solution


def newton_method(
    function: sm.core.expr.Expr,
    symbols: list[sm.core.symbol.Symbol],
    x0: dict[sm.core.symbol.Symbol, float],
    iterations: int = 100,
) -> dict[sm.core.symbol.Symbol, float] or None:
    """
    Perform Newton's method to find the solution to the optimization problem.

    Args:
        function (sm.core.expr.Expr): The objective function to be optimized.
        symbols (list[sm.core.symbol.Symbol]): The symbols used in the objective function.
        x0 (dict[sm.core.symbol.Symbol, float]): The initial values for the symbols.
        iterations (int, optional): The maximum number of iterations. Defaults to 100.

    Returns:
        dict[sm.core.symbol.Symbol, float] or None: The solution to the optimization problem, or None if no solution is found.
    """

    x_star = {}
    x_star[0] = np.array(list(x0.values()))

    # x = [] ## Return x for visual!

    print(f"Starting Values: {x_star[0]}")

    for i in range(iterations):
        # x.append(dict(zip(x0.keys(),x_star[i]))) ## Return x for visual!

        gradient = get_gradient(function, symbols, dict(zip(x0.keys(), x_star[i])))
        hessian = get_hessian(function, symbols, dict(zip(x0.keys(), x_star[i])))

        x_star[i + 1] = x_star[i].T - np.linalg.inv(hessian) @ gradient.T

        if np.linalg.norm(x_star[i + 1] - x_star[i]) < 10e-5:
            solution = dict(zip(x0.keys(), x_star[i + 1]))
            print(f"\nConvergence Achieved ({i+1} iterations): Solution = {solution}")
            break
        else:
            solution = None

        print(f"Step {i+1}: {x_star[i+1]}")

    return solution


def get_gradient(
    function: sm.core.expr.Expr,
    symbols: list[sm.core.symbol.Symbol],
    x0: dict[sm.core.symbol.Symbol, float],
) -> np.ndarray:
    """
    Calculate the gradient of a function at a given point.

    Args:
        function (sm.core.expr.Expr): The function to calculate the gradient of.
        symbols (list[sm.core.symbol.Symbol]): The symbols representing the variables in the function.
        x0 (dict[sm.core.symbol.Symbol, float]): The point at which to calculate the gradient.

    Returns:
        numpy.ndarray: The gradient of the function at the given point.
    """
    d1 = {}
    gradient = np.array([])

    for i in symbols:
        d1[i] = sm.diff(function, i, 1).evalf(subs=x0)
        gradient = np.append(gradient, d1[i])

    return gradient.astype(np.float64)


def get_hessian(
    function: sm.core.expr.Expr,
    symbols: list[sm.core.symbol.Symbol],
    x0: dict[sm.core.symbol.Symbol, float],
) -> np.ndarray:
    """
    Calculate the Hessian matrix of a function at a given point.

    Args:
    function (sm.core.expr.Expr): The function for which the Hessian matrix is calculated.
    symbols (list[sm.core.symbol.Symbol]): The list of symbols used in the function.
    x0 (dict[sm.core.symbol.Symbol, float]): The point at which the Hessian matrix is evaluated.

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


def constrained_newton_method(
    function: sm.core.expr.Expr,
    symbols: list[sm.core.symbol.Symbol],
    x0: dict[sm.core.symbol.Symbol, float],
    iterations: int = 100,
) -> dict[sm.core.symbol.Symbol, float] or None:
    """
    Performs constrained Newton's method to find the optimal solution of a function subject to constraints.

    Parameters:
        function (sm.core.expr.Expr): The function to optimize.
        symbols (list[sm.core.symbol.Symbol]): The symbols used in the function.
        x0 (dict[sm.core.symbol.Symbol, float]): The initial values for the symbols.
        iterations (int, optional): The maximum number of iterations. Defaults to 100.

    Returns:
        dict[sm.core.symbol.Symbol, float] or None: The optimal solution if convergence is achieved, otherwise None.
    """
    x_star = {}
    x_star[0] = np.array(list(x0.values())[:-1])

    optimal_solutions = []
    optimal_solutions.append(dict(zip(list(x0.keys())[:-1], x_star[0])))

    for step in range(iterations):
        # Evaluate function at rho value
        if step == 0:  # starting rho
            rho_sub = list(x0.values())[-1]

        rho_sub_values = {list(x0.keys())[-1]: rho_sub}
        function_eval = function.evalf(subs=rho_sub_values)

        print(f"Step {step} w/ {rho_sub_values}")  # Barrier method step
        print(f"Starting Values: {x_star[0]}")

        # Newton's Method
        for i in range(iterations):
            gradient = get_gradient(
                function_eval, symbols[:-1], dict(zip(list(x0.keys())[:-1], x_star[i]))
            )
            hessian = get_hessian(
                function_eval, symbols[:-1], dict(zip(list(x0.keys())[:-1], x_star[i]))
            )

            x_star[i + 1] = x_star[i].T - np.linalg.inv(hessian) @ gradient.T

            if np.linalg.norm(x_star[i + 1] - x_star[i]) < 10e-5:
                solution = dict(zip(list(x0.keys())[:-1], x_star[i + 1]))
                print(
                    f"Convergence Achieved ({i+1} iterations): Solution = {solution}\n"
                )
                break

        # Record optimal solution & previous optimal solution for each barrier method iteration
        optimal_solution = x_star[i + 1]
        previous_optimal_solution = list(optimal_solutions[step - 1].values())
        optimal_solutions.append(dict(zip(list(x0.keys())[:-1], optimal_solution)))

        # Check for overall convergence
        if np.linalg.norm(optimal_solution - previous_optimal_solution) < 10e-5:
            print(
                f"\n Overall Convergence Achieved ({step} steps): Solution = {optimal_solutions[step]}\n"
            )
            overall_solution = optimal_solutions[step]
            break
        else:
            overall_solution = None

        # Set new starting point
        x_star = {}
        x_star[0] = optimal_solution

        # Update rho
        rho_sub = 0.9 * rho_sub

    return overall_solution
