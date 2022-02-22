import autograd
import autograd.numpy as ag_np
from autograd.scipy.special import expit as ag_logistic_sigmoid

## Define functions for loose hinge bounds

def make_loss_and_grad_for_dataset(
        x_ND, y_N,
        l2_penalty_strength=0.00001,
        lamb=1.0, alpha=0.8):

    # Extract pos and negative examples from dataset    
    N = y_N.size
    bmask_pos_N = y_N==1
    x_pos_ND = x_ND[bmask_pos_N]
    y_pos_N = y_N[bmask_pos_N]

    bmask_neg_N = y_N==0
    x_neg_ND = x_ND[bmask_neg_N]
    y_neg_N = y_N[bmask_neg_N]

    def calc_fp_upper_bound__hinge(w_D, return_per_example_array=False):
        # Apply only to the examples where true y is NEGATIVE
        # Compute real-valued scores via linear function
        u_N = ag_np.dot(x_neg_ND, w_D[:2]) + w_D[2]
        hinge_N = ag_np.maximum(0.0, 1.0 + u_N)
        if return_per_example_array:
            return hinge_N
        return ag_np.sum(hinge_N)


    def calc_tp_lower_bound__hinge(w_D, return_per_example_array=False):
        # Apply only to the examples where true y is POSITIVE
        # Compute real-valued scores via linear function
        u_N = ag_np.dot(x_pos_ND, w_D[:2]) + w_D[2]
        # Use the lower bound on zero one that hinge shape with 1 at + inputs
        neg_hinge_N = ag_np.minimum(1.0, u_N)
        if return_per_example_array:
            return neg_hinge_N
        return ag_np.sum(neg_hinge_N)


    def calc_surrogate_loss(w_D, return_parts=False):
        alpha_ratio = alpha / (1.0 - alpha)
        f = -1.0 * calc_tp_lower_bound__hinge(w_D)
        g = (-1.0 * calc_tp_lower_bound__hinge(w_D)
            + alpha_ratio * calc_fp_upper_bound__hinge(w_D))
        g_or_zero = ag_np.maximum(0.0, g)

        l2_penalty = l2_penalty_strength * ag_np.sum(ag_np.square(w_D[:-1]))

        scaled_loss = (f + l2_penalty + lamb * g_or_zero) / float(N)
        if return_parts:
            return scaled_loss, f, g, l2_penalty
        return scaled_loss

    grad_surrogate_loss = autograd.grad(calc_surrogate_loss)

    return calc_surrogate_loss, grad_surrogate_loss, calc_tp_lower_bound__hinge, calc_fp_upper_bound__hinge

