import numpy as np
import sympy as sm 


def newton_method(function,symbols,x0,iterations=10,mute=False):

    x_star = np.array(list(x0.values()))
    if not mute:
        print(f"Starting Values: {x_star}")

    i=0
    while i < iterations:
        i += 1

        gradient = get_gradient(function, symbols, x0)
        hessian = get_hessian(function, symbols, x0)

        x_star = x_star.T - np.dot(np.linalg.inv(hessian),gradient.T)

        if not mute and i % 5 == 0:
            print(f"Step {i}: {x_star}")

        for key in x0.keys():
            x0[key] = x_star[list(x0.keys()).index(key)]

    return x_star


def get_gradient(function, symbols, x0):

    d1 = {}
    gradient = np.array([])

    for i in symbols:
        d1[i]= sm.diff(function,i,1).evalf(subs=x0)
        gradient = np.append(gradient, d1[i])

    return gradient.astype(np.float64)


def get_hessian(function, symbols, x0):

    d2 = {}
    hessian = np.array([])

    for i in symbols:
        for j in symbols:
            d2[f"{i}{j}"] = sm.diff(function,i,j).evalf(subs=x0)
            hessian = np.append(hessian, d2[f"{i}{j}"])

    hessian = np.array(np.array_split(hessian,len(symbols)))

    return hessian.astype(np.float64)