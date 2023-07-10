import numpy as np

X = np.array(X)

T = 1000
perplexity = 20
η = 100
initialization = 'random'
adaptive_learning = False
n = len(X)

# Compute Pairwise Affinities 
print("Computing Pairwise Affinities....")
σs = []
p_ij = np.zeros(shape=(n,n))
norm = np.zeros(shape=(n,n))
for i in range(0,n):
    diff = X[i]-X
    # Grid search to solve for σ
    p = np.zeros(shape=(n,n))
    def grid_search():
        result = np.inf
        std_norm = np.std(np.linalg.norm(diff, axis=1))
        for σ_search in np.linspace(0.01*std_norm,5*std_norm,500):
            p[i,:] = np.exp(-np.linalg.norm(diff, axis=1)**2/(2*σ_search))
            p[i,i] = 0
            p_new = np.maximum(p[i,:]/np.sum(p[i,:]),1e-9)
            H = np.sum(p_new*np.log2(p_new))
            
            if np.log(perplexity) + H * np.log(2) < result and np.log(perplexity) + H * np.log(2) > 0:
                result = np.log(perplexity) + H * np.log(2)
                σ = σ_search
        
        return σ
    
    σ_i = grid_search()
    σs.append(σ_i)

    p_ij[i,:] = np.exp(-np.linalg.norm(diff, axis=1)**2/(2*σ_i))

    # Set p = 0 when j = i
    np.fill_diagonal(p_ij, 0)

    p_ij[i,:] = p_ij[i,:]/np.sum(p_ij[i,:])

print(f"Mean σ**2 value: {np.mean(σs)}")

# Compute join p_ij
p_ij_master = np.zeros(shape=(n,n))
for i in range(0,n):
    for j in range(0,n):
        p_ij_master[i,j] = (p_ij[i,j] + p_ij[j,i]) / (2*n)

print("Completed Pairwise Affinities. \n")
print("Optimizing Low Dimensional Mapping....")

# Sample Initial Solution 
if initialization == 'random':
    y0 = np.random.randn(n,2)
else:
    X_centered = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_centered)
    y0 = X_centered @ Vt.T[:,:2]

y = np.zeros(shape=(T, n, 2))
y[0] = np.zeros(shape=(n, 2))
y[1] = np.array(y0)

iY = np.zeros((n, 2))
gains = np.ones((n, 2))
min_gain = 0.01
# Main For Loop for High to Low Dimensional Mapping
for t in range(1, T-1):
    
    # Momentum & Early Exaggeration
    if t < 250:
        α = 0.5
        early_exaggeration = 12
    else:
        α = 0.8
        early_exaggeration = 1

    # Compute low-dimensional affinities
    q_ij = np.zeros(shape=(n,n))
    for i in range(0,n):
        diff = y[t][i]-y[t]
        q_ij[i,:] = (1+np.linalg.norm(diff, axis=1)**2)**(-1)

    # Set p = 0 when j = i
    np.fill_diagonal(q_ij, 0)
    
    q_ij = q_ij/q_ij.sum()

    # Ensure no 0 values
    p_ij_master = np.maximum(p_ij_master,10e-8)
    q_ij = np.maximum(q_ij,10e-8)

    # Compute gradient
    gradient = np.zeros(shape=(n, 2))
    for i in range(0,n):
        diff = y[t][i]-y[t]

        A = np.array([(early_exaggeration*p_ij_master[i,:] - q_ij[i,:])])
        B = diff
        C = np.array([(1+np.linalg.norm(diff,axis=1))**(-1)])

        gradient[i] = 4 * np.sum((A * C).T * B, axis=0)

    # Update Rule
    if adaptive_learning:
        gains = (gains + 0.2) * ((gradient > 0.) != (iY > 0.)) + (gains * 0.8) * ((gradient > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = α * iY - η * (gains * gradient)
        y[t+1] = y[t] + iY
        y[t+1] = y[t+1] - np.tile(np.mean(y[t+1], 0), (n, 1))
    else:
        y[t+1] = y[t] - η * gradient + α * (y[t] - y[t-1])

    # Compute current value of cost function
    if t % 50 == 0 or t == 1:
        cost = np.sum(p_ij_master * np.log(p_ij_master / q_ij))
        print(f"Iteration {t}: error is {cost}")

print(f"Completed Low Dimensional Mapping: Final error is {np.sum(p_ij_master * np.log(p_ij_master / q_ij))}")
solution = y[-1]