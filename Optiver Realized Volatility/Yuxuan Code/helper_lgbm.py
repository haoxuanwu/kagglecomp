import lightgbm as lgbm
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

def RMSPE_objective(y_true, y_pred):
    raise NotImplementedError()
    return grad, hess

def RMSPE(y_true, y_pred):
    digits = 5
    err = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))
    return 'RMSPE', np.round(err, digits), False

def R2(y_true, y_pred):
    digits = 5
    return 'R2', np.round(r2_score(y_true, y_pred),digits), True


def run_lgbm(X_train, y_train, eval_set, **lgbm_args):
    # supress eval output
    reg = lgbm.LGBMRegressor(**lgbm_args)
    reg.fit(X_train, y_train,
            eval_set = eval_set,
            eval_names = 'validation',
            eval_metric = ['l2', R2, RMSPE],
            callbacks = [lgbm.log_evaluation(period=0)]
           )
    
    fig, ax = plt.subplots(1, 3, figsize = (10, 4))
    for i,name in enumerate(['l2', 'R2', 'RMSPE']):
        lgbm.plot_metric(reg, ax = ax[i], metric = name, title = name)
    fig.tight_layout()
    
    print('best score')
    print(reg.best_score_['validation'])
    return reg