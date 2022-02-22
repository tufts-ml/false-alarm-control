import numpy as np
import scipy.optimize

from binary_clf_utils import calc_binary_clf_perf, calc_decision_score

def solve_minimimization_multiple_tries_with_lbfgs(
        x_ND, y_N, 
        calc_loss, calc_grad,
        calc_FPUB=None,
        calc_TPLB=None,
        alpha=0.8, gamma=None, delta=None, lamb=None,
        keep_satisifiers_only=False,
        n_inits=1,
        init_w_method='from_uniform',
        init_w_shape=3,
        random_state=101,
        init_w_ID=None,
        verbose=True):

    if isinstance(random_state, int):
        random_state = np.random.RandomState(int(random_state))

    answers = list()

    best_loss = np.inf
    best_w_D = None
    best_issat = None
    
    D = init_w_shape
    if init_w_ID is None:
        init_w_ID = random_state.uniform(low=-3, high=+3, size=D*n_inits)
        init_w_ID = init_w_ID.reshape((n_inits, D))
    elif isinstance(init_w_ID, np.ndarray):
        n_inits = init_w_ID.shape[0]
        

    for ii in range(n_inits):
        init_w_D = init_w_ID[ii]
        ans = scipy.optimize.minimize(
            fun=calc_loss,
            jac=calc_grad,
            x0=init_w_D,
            method='L-BFGS-B')

        w_fmt_msg = D * ' % 6.2f'
        yhat_N = np.asarray(
            calc_decision_score(x_ND, ans.x) >= 0,
            dtype=np.int32)
        perfdict = calc_binary_clf_perf(y_N, yhat_N, gamma=gamma, delta=delta)
        is_sat = 'YES' if perfdict['precision'] >= alpha else 'NO'
        is_sat_bin = 1.0 if perfdict['precision'] >= alpha else 0.0

        summary_msg = "init %02d " + w_fmt_msg + " | loss % 12.5f final w " + w_fmt_msg
        summary_msg = summary_msg % (ii, *init_w_D, ans.fun, *ans.x)
        summary_msg += " | recall %.2f prec %.2f" % (
            perfdict['recall'], perfdict['precision'])
        if keep_satisifiers_only:
            summary_msg += "\nmeets constraint that prec >= %.2f ? %s" % (
                alpha, is_sat)
        
        if verbose:
            print(summary_msg)

        f = None
        g = None
        l2 = None
        if calc_FPUB is not None:
            loss, f, g, l2 = calc_loss(ans.x, return_parts=True)
            fpub = calc_FPUB(ans.x)
            tplb = calc_TPLB(ans.x)
            if 'TP+gamma*delta' in perfdict:
                key = 'TP+gamma*delta'
                label = 'TP+gd'
            else:
                key = 'TP'
                label = 'TP   '
            
            if verbose:
                print("       true %s =%6.1f FP =%6.1f" % (
                    label, perfdict[key],   perfdict['FP']))
                print("  surrogate TP   >=%6.1f FP<=%6.1f" % (tplb, fpub))
                print("  surrogate  f     % .5f  g  % .5f  l2  % .5f" % (f, g, l2))
        if keep_satisifiers_only:
            if is_sat == 'YES':
                cur_loss = -1.0 * perfdict['recall']
            else:
                cur_loss = np.inf
        elif not keep_satisifiers_only:
            cur_loss = ans.fun

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_w_D = ans.x
            best_issat = is_sat_bin

        ans_info = dict(
            w_D=ans.x,
            loss=ans.fun,
            is_sat=is_sat_bin,
            f=f, g=g, l2=l2, lamb=lamb,
            init_id=ii)
        ans_info.update(perfdict)
        answers.append(ans_info)

    # Sort all the answers from best to last
    ranked_best_to_worst_answers = list(sorted(
        answers,
        key=lambda k: k['loss']))
    if best_w_D is None:
        best_answer = ranked_best_to_worst_answers[0]
        best_w_D = best_answer['w_D']
        best_loss = best_answer['loss']
        best_issat = best_answer['is_sat']
    return best_w_D, best_loss, best_issat, ranked_best_to_worst_answers