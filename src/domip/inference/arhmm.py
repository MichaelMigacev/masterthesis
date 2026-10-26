import torch
'''
Formulas as in Murphy SKF paper
"Switching Kalman Filters" Kevin Murphy
https://www.cs.ubc.ca/~murphyk/Papers/skf.pdf

But Notation as in Probabilistic Machine Learning by Murphy
# As explained in "Machine Learning - A Probabilistic Perspective" page 659

# p -- the prior probabilities of the individual domains
# z -- the observation
# u -- the action
# H -- the state transition probabilities for discrete states
# f -- the forward models for the observation
# Q -- covariances for the observation

'''
def update_arhmm(p_bar, z, z_old, f, Q, is_return_likelihoods=False):
    n_models = p_bar.size()[0]
    p_hat = torch.zeros_like(p_bar)
    likelihood = torch.zeros_like(p_bar)
    for i in range(n_models):
        innovation = z - f[i](z_old)
        nu = innovation
        # eq 17
        #F = torch.autograd.functional.jacobian(f[i], z_old, create_graph=False) # not sure if z_old here, could also use z
        #S = torch.mm(torch.mm(F, Sigma_bar[i]), F.transpose(0, 1)) + Q[i](z_old) # copied and adapted from my ekf implementation
        S = Q[i] # copied and adapted from my ekf implementation

        # eq 15 (without the normalization)
        likelihood[i] = torch.det(S)**-.5 * torch.exp(-.5*nu.dot(torch.inverse(S).mv(nu)))
        p_hat[i] = p_bar[i] * likelihood[i]
    p_hat /= torch.sum(p_hat)
    if is_return_likelihoods:
        return p_hat, likelihood
    else:
        return p_hat

# p -- the prior probabilities of the individual domains
# z -- the observation
# H -- the state transition probabilities for discrete states
# h -- observation models for the discrete states
# R -- covariance for the observation model

def predict_arhmm(p, H):
    p_bar = H.mv(p)
    p_bar = p_bar / torch.sum(p_bar)
    return p_bar

def forward_backward_arhmm(p, zs, fs, Q, H):
    # forward
    ps = torch.zeros(T,2)
    likelihoods = torch.zeros(T,2)
    ps[0] = p
    for i in range(1,T):
        p_bar = predict_arhmm(ps[0], H)
        ps[i], likelihoods[i] = update_arhmm(p_bar, zs[i], zs[i-1], fs, Q, is_return_likelihoods=True)


    # backward
    beta = torch.ones(T, 2)
    for i in range(T-2, 0, -1):
        beta[i] = H.mv(likelihoods[i+1]*beta[i+1])

    # print(ps)
    # print(beta)
    ps_bar = ps * beta
    # print(ps_bar)
    # print(ps_bar.sum(dim=1))
    return ps_bar / ps_bar.sum(dim=1).unsqueeze(1)

def em_arhmm(zs, K, n_iter=3):
    f = [lambda x:x]
    for iteration in range(n_iter):
        p = torch.tensor([.5, .5])
        # Expect
        ps = forward_backward_arhmm(p, zs, f, Q, H)
        # maximize


if __name__ == "__main__":

    dim_observation = 2
    p_init = torch.tensor([.5, .5])
    H = torch.tensor([[.9, .1],
                      [.1, .9]])

    z_mu = torch.tensor([[0., 2.1], [0., .9]])
    R = torch.tensor([[[10.1, 0.],
                       [0., 10.1]],
                      [[10.1, 0.],
                       [0., 10.1]]])

    fs = [lambda x:x,
         lambda x:x*2]

    # create input data
    T = 20
    s = torch.zeros(T, dtype=torch.long)
    zs = torch.zeros(T,dim_observation)
    distrib = torch.distributions.MultivariateNormal(loc=z_mu[0], covariance_matrix=R[0])
    zs[0] = distrib.sample()
    for t in range(1,20):
        categorical = torch.distributions.Categorical(H[s[t-1]])

        s[t] = categorical.sample((1,))
        distrib = torch.distributions.MultivariateNormal(loc=fs[s[t]](zs[t-1]), covariance_matrix=R[s[t]])
        zs[t] = distrib.sample()

    print(zs.T)
    print(s)
    ps = torch.zeros(T,2)
    p = p_init
    for i in range(1,T):

        p = predict_arhmm(p, H)
        # print(p)
        p = update_arhmm(p, zs[i], zs[i-1], fs, R)
        # print(p)
        ps[i] = p
    torch.set_printoptions(precision=2)
    print("forward:", ps[:,1].round().int())
    # print("forward:", ps[:,1])
    ps = forward_backward_arhmm(p_init, zs, fs, R, H)
    print("forback:", ps[:,1].round().int())
    # print("forback:", ps[:,1])