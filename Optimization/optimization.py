import numpy as np
import sympy as sm 



def gradient_descent(function,symbols,x0,learning_rate=0.1,iterations=100,mute=False):

    x_star = {}
    x_star[0] = np.array(list(x0.values()))

    x = []

    if not mute:
        print(f"Starting Values: {x_star[0]}")

    i=0
    while i < iterations:

        x.append(dict(zip(x0.keys(),x_star[i])))

        gradient = get_gradient(function, symbols, dict(zip(x0.keys(),x_star[i])))

        x_star[i+1] = x_star[i].T - learning_rate*gradient.T

        if np.linalg.norm(x_star[i+1] - x_star[i]) < 10e-7 and i != 1:
            print(f"\nConvergence Achieved ({i+1} iterations): Solution = {dict(zip(x0.keys(),x_star[i+1]))}")
            break 

        if not mute:
            print(f"Step {i+1}: {x_star[i+1]}")

        i += 1
        
    return x

    

def newton_method(function,symbols,x0,iterations=100,mute=False):

    x_star = {}
    x_star[0] = np.array(list(x0.values()))
    
    x = []

    if not mute:
        print(f"Starting Values: {x_star[0]}")

    i=0
    while i < iterations:

        x.append(dict(zip(x0.keys(),x_star[i])))

        gradient = get_gradient(function, symbols, dict(zip(x0.keys(),x_star[i])))
        hessian = get_hessian(function, symbols, dict(zip(x0.keys(),x_star[i])))

        x_star[i+1] = x_star[i].T - np.dot(np.linalg.inv(hessian),gradient.T)

        if np.linalg.norm(x_star[i+1] - x_star[i]) < 10e-7 and i != 1:
            print(f"\nConvergence Achieved ({i+1} iterations): Solution = {dict(zip(x0.keys(),x_star[i+1]))}")
            break

        if not mute:
            print(f"Step {i+1}: {x_star[i+1]}")

        i += 1

    return x



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



def constrained_newton_method(function,symbols,x0,iterations=10000,mute=False):

    x_star = {}
    x_star[0] = np.array(list(x0.values())[:-1])

    optimal_solutions = []
    optimal_solutions.append(dict(zip(list(x0.keys())[:-1],x_star[0])))

    step = 1 
    while True:
        
        # Evaluate function at rho value
        if step == 1: # starting rho
            rho_sub = list(x0.values())[-1]

        rho_sub_values = {list(x0.keys())[-1]:rho_sub}
        function_eval = function.evalf(subs=rho_sub_values)

        if not mute:
                print(f"Step {step} w/ {rho_sub_values}") # Barrier method step
                print(f"Starting Values: {x_star[0]}")
        
        # Newton's Method
        i=0
        while i < iterations:
            i += 1
        
            gradient = get_gradient(function_eval, symbols[:-1], dict(zip(list(x0.keys())[:-1],x_star[i-1])))
            hessian = get_hessian(function_eval, symbols[:-1], dict(zip(list(x0.keys())[:-1],x_star[i-1])))

            x_star[i] = x_star[i-1].T - np.dot(np.linalg.inv(hessian),gradient.T)

            if np.linalg.norm(x_star[i] - x_star[i-1]) < 10e-5 and i != 1:
                print(f"Convergence Achieved ({i} iterations): Solution = {dict(zip(list(x0.keys())[:-1],x_star[i]))}\n") 
                break
        
        # Record optimal solution for each barrier method iteration
        optimal_solution = x_star[i]
        optimal_solutions.append(dict(zip(list(x0.keys())[:-1],optimal_solution)))
        
        # Check for overall convergence
        previous_optimal_solution = list(optimal_solutions[step-2].values())
        if step != 1 and np.linalg.norm(optimal_solution - previous_optimal_solution) < 10e-5:
            print(f"\n Overall Convergence Achieved ({step} steps): Solution = {dict(zip(list(x0.keys())[:-1],optimal_solution))}\n")
            break

        # Set new starting point
        x_star = {}
        x_star[0] = optimal_solution

        # Update rho
        rho_sub = 0.9*rho_sub

        # Update Steps
        step += 1

    return optimal_solutions






#####################
#  Work In-Progress #
#####################

def quasi_newton_method(function,symbols,x0,method='modified',iterations=10, mute=False):

    x_star = {}
    x_star[0] = np.array(list(x0.values()))
    
    if not mute:
        print(f"Starting Values: {x_star[0]}")

    G = np.identity(len(symbols),dtype=np.float64)

    i=0
    while i < iterations:
        i += 1

        gradient = get_gradient(function, symbols, x0)

        if method == "modified":
            G = get_hessian(function, symbols, x0)
            u = -np.dot(np.linalg.inv(G),gradient.T)
        else:
            u = -np.dot(G,gradient.T)

        L = get_lambda(function, x_star[i-1], x0, u)
        
        x_star[i] = x_star[i-1].T + L*u

        if i % 5 == 0 or i == 1:
            print(f"Step {i}: {x_star[i]}")
        
        if np.linalg.norm(x_star[i] - x_star[i-1]) < 10e-10 and i != 1:
            return print(f"Convergence Achieved: {x_star[i]}")
        
        x0 = dict(zip(x0.keys(),x_star[i]))

        # Update Hessian 
        v = L*u
        y = get_gradient(function, symbols, x0).T - gradient.T

        if method=="DFP":
            G = rank1update(G, v, y)
        elif method=="BFGS":
            G = rank2update(G,v,y)
        elif method=="modified":
            pass
        else:
            raise ValueError('Choose between DFP & BFGS for methods.')

    return


def get_lambda(function, x_star, x0, u):
    
    L = sm.symbols('L')

    line_search = function.evalf(subs={k:v for (k,v) in zip(x0.keys(),x_star.T + L*u)})

    lambdaStar = newton_method(line_search,[L],{L:0},20,mute=True)

    return lambdaStar
    

def rank1update(G, v, y):

    A = np.dot(v,v.T)/np.dot(v.T,y)
    B = - np.dot(np.dot(G,y),np.dot(G,y).T)/np.dot(np.dot(y.T,G),y)
    G = G + A + B

    return G


def rank2update(G, v, y):

    A = (1 + (np.dot(np.dot(y.T,G),y)/np.dot(v.T,y)))*(np.dot(v,v.T)/np.dot(v.T,y))
    B = - (np.dot(np.dot(v,y.T),G)+np.dot(np.dot(G,y),v.T))/np.dot(v.T,y)
    G = G + A + B
    
    return G