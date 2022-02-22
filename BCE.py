import autograd
import autograd.numpy as ag_np
from autograd.scipy.special import expit as ag_logistic_sigmoid

def make_loss_and_grad_for_dataset(x_ND, y_N):

	N = y_N.size

	def calc_cross_entropy_loss(w_D):  
	    # Convert y_N to be either 1 or -1
	    ry_N = ag_np.sign(y_N-0.01)

	    # Compute real-valued scores fed into sigmoid
	    u_N = ry_N * (ag_np.dot(x_ND, w_D[:2]) + w_D[2]) 

	    # Add 1e-15 to avoid precision problems
	    proba1_N = ag_logistic_sigmoid(u_N)

	    bce = -1.0 * ag_np.sum(ag_np.log(proba1_N + 1e-15)) 
	    return bce / ag_np.maximum(N, 1.0)

	grad_cross_entropy_loss = autograd.grad(calc_cross_entropy_loss)

	return calc_cross_entropy_loss, grad_cross_entropy_loss
