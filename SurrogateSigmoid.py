import autograd
import autograd.numpy as ag_np
from autograd.scipy.special import expit as ag_logistic_sigmoid

import scipy.optimize

## Functions for creating sigmoid bounds given gamma, delta, epsilon
def calc_s(r, m, b, gamma=2.0, delta=.01):
    return (1.0 + gamma*delta) * ag_logistic_sigmoid(m * r + b)

def make_loss_and_grad_FPUB(gamma, delta, epsilon):
    def calc_loss(mb_vec):
        m = mb_vec[0]
        b = mb_vec[1]
        return (
            ag_np.square(delta - calc_s(-epsilon, m, b, gamma, delta))
            + ag_np.square(1.0 + delta - calc_s(0.0, m, b, gamma, delta))
        )
    calc_grad = autograd.grad(calc_loss)
    return calc_loss, calc_grad

def make_calc_sigmoid_FPUB(gamma, delta, epsilon):
    calc_loss, calc_grad = make_loss_and_grad_FPUB(gamma, delta, epsilon)
    ans = scipy.optimize.minimize(
        fun=calc_loss,
        jac=calc_grad,
        x0=ag_np.asarray([0.0, 0.0]),
        options=dict(ftol=1e-13, gtol=0.0),
        method='L-BFGS-B')
    m = ans.x[0]
    b = ans.x[1]
    def calc_sigmoid(r):
        return (1 + gamma * delta) * ag_logistic_sigmoid(r * m + b)
    return calc_sigmoid, m, b

def make_loss_and_grad_TPLB(gamma, delta, epsilon):
    def calc_loss(mb_vec):
        m = mb_vec[0]
        b = mb_vec[1]
        return (
            ag_np.square(delta - calc_s(0.0, m, b, gamma, delta))
            + ag_np.square(1.0 + delta - calc_s(epsilon, m, b, gamma, delta))
        )
    calc_grad = autograd.grad(calc_loss)
    return calc_loss, calc_grad

def make_calc_sigmoid_TPLB(gamma, delta, epsilon):
    calc_loss, calc_grad = make_loss_and_grad_TPLB(gamma, delta, epsilon)
    ans = scipy.optimize.minimize(
        fun=calc_loss,
        jac=calc_grad,
        x0=ag_np.asarray([0.0, 0.0]),
        options=dict(ftol=1e-13, gtol=0.0),
        method='L-BFGS-B')
    def calc_sigmoid(r):
        return (1 + gamma * delta) * ag_logistic_sigmoid(r * ans.x[0] + ans.x[1])
    m = ans.x[0]
    b = ans.x[1]
    return calc_sigmoid, m, b


def make_loss_and_grad_for_dataset(
        x_ND, y_N,
        lamb=1.0, alpha=0.8,
        l2_penalty_strength=0.00001,
        gamma=4.0, delta=0.05, epsilon=0.8, verbose=True, return_scale_corrected_loss=False):

    # Extract pos and negative examples from dataset    
    N = y_N.size
    bmask_pos_N = y_N==1
    x_pos_ND = x_ND[bmask_pos_N]
    y_pos_N = y_N[bmask_pos_N]

    bmask_neg_N = y_N==0
    x_neg_ND = x_ND[bmask_neg_N]
    y_neg_N = y_N[bmask_neg_N]

    # Create custom slope/intercept for each bound
    calc_sigmoid_FPUB, m_FPUB, b_FPUB = make_calc_sigmoid_FPUB(
        gamma, delta, epsilon)
    calc_sigmoid_TPLB, m_TPLB, b_TPLB = make_calc_sigmoid_TPLB(
        gamma, delta, epsilon)
    
    if verbose:
        print("FPUB")
        print("slope     % .7f" % m_FPUB)
        print("intercept % .7f" % b_FPUB)

        print("TPLB")
        print("slope     % .7f" % m_TPLB)
        print("intercept % .7f" % b_TPLB)

    def calc_fp_upper_bound__sigmoid(w_D, return_per_example_array=False):
        # Apply only to the examples where true y is NEGATIVE
        # Compute real-valued scores via linear function
        u_N = ag_np.dot(x_neg_ND, w_D[:2]) + w_D[2]
        # Evaluate bound at each point and return sum
        p_N = calc_sigmoid_FPUB(u_N)
        if return_per_example_array:
            return p_N
        return ag_np.sum(p_N)

    def calc_tp_lower_bound__sigmoid(w_D, return_per_example_array=False):
        # Apply only to the examples where true y is POSITIVE
        # Compute real-valued scores via linear function
        u_N = ag_np.dot(x_pos_ND, w_D[:2]) + w_D[2]
        # Evaluate bound at each point and return sum
        p_N = calc_sigmoid_TPLB(u_N)
        if return_per_example_array:
            return p_N
        return ag_np.sum(p_N)

    def calc_surrogate_loss(w_D, return_parts=False):
        alpha_ratio = alpha / (1.0 - alpha)
        f = -1.0 * calc_tp_lower_bound__sigmoid(w_D)
        g = (-1.0 * calc_tp_lower_bound__sigmoid(w_D)
            + alpha_ratio * calc_fp_upper_bound__sigmoid(w_D))
        g_or_zero = ag_np.maximum(0.0, g)

        l2_penalty = l2_penalty_strength * ag_np.sum(ag_np.square(w_D[:-1]))

        scaled_loss = (f + l2_penalty + lamb * g_or_zero) / float(N)
        if return_parts:
            return scaled_loss, f, g, l2_penalty
        return scaled_loss
    
    def calc_surrogate_loss_with_scale_correction(w_D, return_parts=False):
        alpha_ratio = alpha / (1.0 - alpha)
        N_pos = len(y_pos_N)
        tpc_scale_correction = gamma*delta*N_pos
        f = -1.0 * calc_tp_lower_bound__sigmoid(w_D)
        g = (-1.0 * calc_tp_lower_bound__sigmoid(w_D)
            + alpha_ratio * calc_fp_upper_bound__sigmoid(w_D)+tpc_scale_correction)
        g_or_zero = ag_np.maximum(0.0, g)

        l2_penalty = l2_penalty_strength * ag_np.sum(ag_np.square(w_D[:-1]))

        scaled_loss = (f + l2_penalty + lamb * g_or_zero) / float(N)
        if return_parts:
            return scaled_loss, f, g, l2_penalty
        return scaled_loss
    
    grad_surrogate_loss = autograd.grad(calc_surrogate_loss)
    grad_surrogate_loss_with_scale_correction = autograd.grad(calc_surrogate_loss_with_scale_correction)
    
    if return_scale_corrected_loss:
        return calc_surrogate_loss_with_scale_correction, grad_surrogate_loss_with_scale_correction, calc_tp_lower_bound__sigmoid, calc_fp_upper_bound__sigmoid
    else:
        return calc_surrogate_loss, grad_surrogate_loss, calc_tp_lower_bound__sigmoid, calc_fp_upper_bound__sigmoid

