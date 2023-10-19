import numpy as np
import warnings
import math

warnings.filterwarnings("ignore")


def grid_search(diff_i, i, perplexity):

    '''
    Helper function to obtain σ's based on user-specified perplexity.
    '''

    result = np.inf # Set first result to be infinity

    norm = np.linalg.norm(diff_i, axis=1)
    std_norm = np.std(norm) # Use standard deviation of norms to define search space

    for σ_search in np.linspace(0.01*std_norm,5*std_norm,200):

        # Equation 1 Numerator
        p = np.exp(-norm**2/(2*σ_search**2)) 

        # Set p = 0 when i = j
        p[i] = 0 

        # Equation 1 (ε -> 0) 
        ε = np.nextafter(0,1)
        p_new = np.maximum(p/np.sum(p),ε)
        
        # Shannon Entropy
        H = -np.sum(p_new*np.log2(p_new))
        
        # Get log(perplexity equation) as close to equality
        if np.abs(np.log(perplexity) - H * np.log(2)) < np.abs(result):
            result = np.log(perplexity) - H * np.log(2)
            σ = σ_search
    
    return σ


def get_original_pairwise_affinities(X:np.array([]), 
                                     perplexity=10):

    '''
    Function to obtain affinities matrix.
    '''

    n = len(X)

    print("Computing Pairwise Affinities....")

    p_ij = np.zeros(shape=(n,n))
    for i in range(0,n):
        
        # Equation 1 numerator
        diff = X[i]-X
        σ_i = grid_search(diff, i, perplexity) # Grid Search for σ_i
        norm = np.linalg.norm(diff, axis=1)
        p_ij[i,:] = np.exp(-norm**2/(2*σ_i**2))

        # Set p = 0 when j = i
        np.fill_diagonal(p_ij, 0)
        
        # Equation 1 
        p_ij[i,:] = p_ij[i,:]/np.sum(p_ij[i,:])

    # Set 0 values to minimum numpy value (ε approx. = 0) 
    ε = np.nextafter(0,1)
    p_ij = np.maximum(p_ij,ε)

    print("Completed Pairwise Affinities Matrix. \n")

    return p_ij


def get_symmetric_p_ij(p_ij:np.array([])):

    '''
    Function to obtain symmetric affinities matrix utilized in t-SNE.
    '''
        
    print("Computing Symmetric p_ij matrix....")

    n = len(p_ij)
    p_ij_symmetric = np.zeros(shape=(n,n))
    for i in range(0,n):
        for j in range(0,n):
            p_ij_symmetric[i,j] = (p_ij[i,j] + p_ij[j,i]) / (2*n)
    
    # Set 0 values to minimum numpy value (ε approx. = 0)
    ε = np.nextafter(0,1)
    p_ij_symmetric = np.maximum(p_ij_symmetric,ε)

    print("Completed Symmetric p_ij Matrix. \n")

    return p_ij_symmetric


def initialization(X: np.array([]),
                   n_dimensions = 2,
                   initialization = 'random'):

    '''
    Obtain initial solution for t-SNE either randomly or using PCA.
    '''

    # Sample Initial Solution 
    if initialization == 'random' or initialization != 'PCA':
        y0 = np.random.normal(loc=0,scale=1e-4,size=(len(X),n_dimensions))
    elif initialization == 'PCA':
        X_centered = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered)
        y0 = X_centered @ Vt.T[:,:n_dimensions]

    return y0


def get_low_dimensional_affinities(Y:np.array([])):
    '''
    Obtain low-dimensional affinities.
    '''

    n = len(Y)
    q_ij = np.zeros(shape=(n,n))

    for i in range(0,n):

        # Equation 4 Numerator
        diff = Y[i]-Y
        norm = np.linalg.norm(diff, axis=1)
        q_ij[i,:] = (1+norm**2)**(-1)

    # Set p = 0 when j = i
    np.fill_diagonal(q_ij, 0)

    # Equation 4 
    q_ij = q_ij/q_ij.sum()

    # Set 0 values to minimum numpy value (ε approx. = 0)
    ε = np.nextafter(0,1)
    q_ij = np.maximum(q_ij,ε)

    return q_ij


def get_gradient(p_ij: np.array([]),
                q_ij: np.array([]),
                Y: np.array([])):
    '''
    Obtain gradient of cost function at current point Y.
    '''

    n = len(p_ij)

    # Compute gradient
    gradient = np.zeros(shape=(n, Y.shape[1]))
    for i in range(0,n):

        # Equation 5
        diff = Y[i]-Y
        A = np.array([(p_ij[i,:] - q_ij[i,:])])
        B = np.array([(1+np.linalg.norm(diff,axis=1))**(-1)])
        C = diff
        gradient[i] = 4 * np.sum((A * B).T * C, axis=0)

    return gradient


def low_dimensional_embedding(p_ij: np.array([]), 
                              Y0: np.array([]),
                              T = 1000, 
                              η = 200,
                              early_exaggeration = 4,
                              n_dimensions = 2):
    
    print("Optimizing Low Dimensional Embedding....")
    
    n = len(p_ij)

    # Initializations
    Y = np.zeros(shape=(T, n, n_dimensions))
    Y[0] = np.zeros(shape=(n, n_dimensions))
    Y[1] = np.array(Y0)

    # Main For Loop for High to Low Dimensional Mapping
    for t in range(1, T-1):
        
        # Momentum & Early Exaggeration
        if t < 250:
            α = 0.5
            early_exaggeration = early_exaggeration
        else:
            α = 0.8
            early_exaggeration = 1

        # Get Low Dimensional Affinities
        q_ij = get_low_dimensional_affinities(Y[t])

        # Get Gradient of Cost Function
        gradient = get_gradient(early_exaggeration*p_ij, q_ij, Y[t])

        # Update Rule
        Y[t+1] = Y[t] - η * gradient + α * (Y[t] - Y[t-1]) # Minimizing Cost Function

        # Compute current value of cost function
        if t % 50 == 0 or t == 1:
            cost = np.sum(p_ij * np.log(p_ij / q_ij))
            print(f"Iteration {t}: error is {cost}")

    print(f"Completed Low Dimensional Embedding: Final error is {np.sum(p_ij * np.log(p_ij / q_ij))}")
    solution = Y[-1]

    return solution, Y



def tsne(X: np.array([]), 
        perplexity = 10,
        T = 1000, 
        η = 200,
        early_exaggeration = 4,
        n_dimensions = 2):
    
    n = len(X)

    # Get original affinities matrix 
    p_ij = get_original_pairwise_affinities(X, perplexity)
    p_ij_symmetric = get_symmetric_p_ij(p_ij)
    
    # Initialization
    Y = np.zeros(shape=(T, n, n_dimensions))
    Y_minus1 = np.zeros(shape=(n, n_dimensions))
    Y[0] = Y_minus1
    Y1 = initialization(X, n_dimensions)
    Y[1] = np.array(Y1)

    print("Optimizing Low Dimensional Embedding....")
    # Optimization
    for t in range(1, T-1):
        
        # Momentum & Early Exaggeration
        if t < 250:
            α = 0.5
            early_exaggeration = early_exaggeration
        else:
            α = 0.8
            early_exaggeration = 1

        # Get Low Dimensional Affinities
        q_ij = get_low_dimensional_affinities(Y[t])

        # Get Gradient of Cost Function
        gradient = get_gradient(early_exaggeration*p_ij_symmetric, q_ij, Y[t])

        # Update Rule
        Y[t+1] = Y[t] - η * gradient + α * (Y[t] - Y[t-1]) # Use negative gradient 

        # Compute current value of cost function
        if t % 50 == 0 or t == 1:
            cost = np.sum(p_ij_symmetric * np.log(p_ij_symmetric / q_ij))
            print(f"Iteration {t}: Value of Cost Function is {cost}")

    print(f"Completed Low Dimensional Embedding: Final Value of Cost Function is {np.sum(p_ij_symmetric * np.log(p_ij_symmetric / q_ij))}")
    solution = Y[-1]

    return solution, Y