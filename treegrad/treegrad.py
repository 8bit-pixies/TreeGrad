"""
Tree Grad

Implementation of an online learning approach for tree based models
"""
import lightgbm as lgb
import copy

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer

import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from autograd.misc.flatten import flatten as weights_flatten

from treegrad.tree_utils import split_trees_by_classes, multiclass_trees_to_param, gbm_gen, simple_callback, multi_tree_to_param, sigmoid, generate_batch

class BaseTreeGrad(BaseEstimator):
    def __init__(self, num_leaves=31, max_depth=-1, 
                 learning_rate=0.1, n_estimators=100,
                 autograd_config={'refit_splits':False, 'batch_size': 32}
    ):
        self.ensemble_config = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators
        }
        self.autograd_config = autograd_config

class TGDClassifier(BaseTreeGrad, ClassifierMixin):
    def fit(self, X, y):
        self.base_model_ = lgb.LGBMClassifier(**self.ensemble_config)
        self.base_model_.fit(X, y)
        self.n_classes_ = self.base_model_.n_classes_
        self.is_partial = False
        return self
    
    def partial_fit_base(self, X, y):
        check_is_fitted(self, 'base_model_')

        batch_indices = generate_batch(X, self.autograd_config.get('batch_size', 32))

        esp = 1e-11 # where should this live?
        step_size = self.autograd_config.get('step_size', 0.05)
        callback = None if self.autograd_config.get('verbose', False) else simple_callback
        num_iters = self.autograd_config.get('num_iters', 1000)

        nclass = self.n_classes_
        model_dump = self.base_model_.booster_.dump_model()
        trees_ = [m["tree_structure"] for m in model_dump["tree_info"]]

        if nclass == 2:
            y_ohe = y
        else:
            y_ohe = LabelBinarizer().fit_transform(y)

        if nclass > 2:
            trees = split_trees_by_classes(trees_, nclass)
            trees_params = multiclass_trees_to_param(X, y, trees)
            model_ = gbm_gen(trees_params[0], X, trees_params[2], trees_params[1], True, nclass)

            def training_loss(weights, idx=0):
                # Training loss is the negative log-likelihood of the training labels.
                t_idx_ = batch_indices(idx)
                preds = model_(weights, X[t_idx_, :])
                loglik = -np.sum(np.log(preds+esp) * y_ohe[t_idx_, :])

                num_unpack = 3
                reg = 0
                # reg_l1 = np.sum(np.abs(flattened)) * 1.
                for idx_ in range(0, len(weights), num_unpack):
                    param_temp_ = weights[idx_:idx_+num_unpack]
                    flattened, _ = weights_flatten(param_temp_[:2])
                    reg_l1 = np.sum(np.abs(flattened)) * 1.
                    reg += reg_l1
                return loglik + reg

        else:
            trees_params = multi_tree_to_param(X, y, trees_)
            model_ = gbm_gen(trees_params[0], X, trees_params[2], trees_params[1], False, 2)

            def training_loss(weights, idx=0):
                # Training loss is the negative log-likelihood of the training labels.
                t_idx_ = batch_indices(idx)
                preds = sigmoid(model_(weights, X[t_idx_, :]))
                label_probabilities = preds * y[t_idx_] + (1 - preds) * (1 - y[t_idx_])
                #print(label_probabilities)
                loglik = -np.sum(np.log(label_probabilities))

                num_unpack = 3
                reg = 0
                # reg_l1 = np.sum(np.abs(flattened)) * 1.
                for idx_ in range(0, len(weights), num_unpack):
                    param_temp_ = weights[idx_:idx_+num_unpack]
                    flattened, _ = weights_flatten(param_temp_[:2])
                    reg_l1 = np.sum(np.abs(flattened)) * 1.
                    reg += reg_l1
                return loglik + reg

        training_gradient_fun = grad(training_loss)
        param_ = adam(training_gradient_fun, trees_params[0], callback=callback, step_size=step_size, num_iters=num_iters)

        self.base_param_ = copy.deepcopy(trees_params)
        self.partial_param_ = param_
        self.is_partial = True
        return self


    def partial_fit_param(self, X, y):
        check_is_fitted(self, 'base_model_')
        check_is_fitted(self, 'base_param_')
        check_is_fitted(self, 'partial_param_')

        batch_indices = generate_batch(X, self.autograd_config.get('batch_size', 32))

        esp = 1e-11 # where should this live?
        step_size = self.autograd_config.get('step_size', 0.05)
        callback = None if self.autograd_config.get('verbose', False) else simple_callback
        num_iters = self.autograd_config.get('num_iters', 1000)
        nclass = self.n_classes_

        if nclass == 2:
            y_ohe = y
        else:
            y_ohe = LabelBinarizer().fit_transform(y)

        if nclass > 2:
            model_ = gbm_gen(self.base_param_[0], X, self.base_param_[2], self.base_param_[1], True, nclass)

            def training_loss(weights, idx=0):
                # Training loss is the negative log-likelihood of the training labels.
                t_idx_ = batch_indices(idx)
                preds = model_(weights, X[t_idx_, :])
                loglik = -np.sum(np.log(preds+esp) * y_ohe[t_idx_, :])

                num_unpack = 3
                reg = 0
                # reg_l1 = np.sum(np.abs(flattened)) * 1.
                for idx_ in range(0, len(weights), num_unpack):
                    param_temp_ = weights[idx_:idx_+num_unpack]
                    flattened, _ = weights_flatten(param_temp_[:2])
                    reg_l1 = np.sum(np.abs(flattened)) * 1.
                    reg += reg_l1
                return loglik + reg

        else:
            model_ = gbm_gen(self.base_param_[0], X, self.base_param_[2], self.base_param_[1], False, 2)

            def training_loss(weights, idx=0):
                # Training loss is the negative log-likelihood of the training labels.
                t_idx_ = batch_indices(idx)
                preds = sigmoid(model_(weights, X[t_idx_, :]))
                label_probabilities = preds * y[t_idx_] + (1 - preds) * (1 - y[t_idx_])
                #print(label_probabilities)
                loglik = -np.sum(np.log(label_probabilities))

                num_unpack = 3
                reg = 0
                # reg_l1 = np.sum(np.abs(flattened)) * 1.
                for idx_ in range(0, len(weights), num_unpack):
                    param_temp_ = weights[idx_:idx_+num_unpack]
                    flattened, _ = weights_flatten(param_temp_[:2])
                    reg_l1 = np.sum(np.abs(flattened)) * 1.
                    reg += reg_l1
                return loglik + reg

        training_gradient_fun = grad(training_loss)
        param_ = adam(training_gradient_fun, self.partial_param_, callback=callback, step_size=step_size, num_iters=num_iters)

        self.partial_param_ = param_
        self.is_partial = True
        return self

    def partial_fit(self, X, y):
        check_is_fitted(self, 'base_model_')
        if self.is_partial:
            self.partial_fit_param(X, y)
        else:
            self.partial_fit_base(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, 'base_model_')
        if not self.is_partial:
            return self.base_model_.predict(X)
        else:
            multi_class = self.n_classes_ > 2
            model_ = gbm_gen(self.partial_param_, X, self.base_param_[2], self.base_param_[1], multi_class, self.n_classes_)
            preds = model_(self.partial_param_, X)
            if self.n_classes_ > 2:
                return np.argmax(preds, axis=1)
            else:
                return np.round(sigmoid(preds))

    def predict_proba(self, X):
        check_is_fitted(self, 'base_model_')
        if not self.is_partial:
            return self.base_model_.predict_proba(X)
        else:
            multi_class = self.n_classes_ > 2
            model_ = gbm_gen(self.partial_param_, X, self.base_param_[2], self.base_param_[1], multi_class, self.n_classes_)
            preds = model_(self.partial_param_, X)
            if self.n_classes_ > 2:
                return preds
            else:
                pred_positive = sigmoid(preds)
                return np.stack([1-pred_positive, pred_positive], axis=-1)




class TGDRegressor(BaseTreeGrad, RegressorMixin):
    def fit(self, X, y):
        self.base_model_ = lgb.LGBMRegressor(**self.ensemble_config)
        self.base_model_.fit(X, y)
        self.n_classes_ = 1
        self.is_partial = False
        return self
    
    def partial_fit_base(self, X, y):
        check_is_fitted(self, 'base_model_')

        batch_indices = generate_batch(X, self.autograd_config.get('batch_size', 32))

        esp = 1e-11 # where should this live?
        step_size = self.autograd_config.get('step_size', 0.05)
        callback = None if self.autograd_config.get('verbose', False) else simple_callback
        num_iters = self.autograd_config.get('num_iters', 1000)

        nclass = self.n_classes_
        model_dump = self.base_model_.booster_.dump_model()
        trees_ = [m["tree_structure"] for m in model_dump["tree_info"]]

    
        trees_params = multi_tree_to_param(X, y, trees_)
        model_ = gbm_gen(trees_params[0], X, trees_params[2], trees_params[1], False, 2)

        def training_loss(weights, idx=0):
            # Training loss is the negative log-likelihood of the training labels.
            t_idx_ = batch_indices(idx)
            preds = sigmoid(model_(weights, X[t_idx_, :]))
            label_probabilities = preds * y[t_idx_] + (1 - preds) * (1 - y[t_idx_])
            #print(label_probabilities)
            loglik = -np.sum(np.log(label_probabilities))

            num_unpack = 3
            reg = 0
            # reg_l1 = np.sum(np.abs(flattened)) * 1.
            for idx_ in range(0, len(weights), num_unpack):
                param_temp_ = weights[idx_:idx_+num_unpack]
                flattened, _ = weights_flatten(param_temp_[:2])
                reg_l1 = np.sum(np.abs(flattened)) * 1.
                reg += reg_l1
            return loglik + reg

        training_gradient_fun = grad(training_loss)
        param_ = adam(training_gradient_fun, trees_params[0], callback=callback, step_size=step_size, num_iters=num_iters)

        self.base_param_ = copy.deepcopy(trees_params)
        self.partial_param_ = param_
        self.is_partial = True
        return self


    def partial_fit_param(self, X, y):
        check_is_fitted(self, 'base_model_')
        check_is_fitted(self, 'base_param_')
        check_is_fitted(self, 'partial_param_')

        batch_indices = generate_batch(X, self.autograd_config.get('batch_size', 32))

        esp = 1e-11 # where should this live?
        step_size = self.autograd_config.get('step_size', 0.05)
        callback = None if self.autograd_config.get('verbose', False) else simple_callback
        num_iters = self.autograd_config.get('num_iters', 1000)
        nclass = self.n_classes_

        model_ = gbm_gen(self.base_param_[0], X, self.base_param_[2], self.base_param_[1], False, 2)

        def training_loss(weights, idx=0):
            # Training loss is the negative log-likelihood of the training labels.
            t_idx_ = batch_indices(idx)
            preds = sigmoid(model_(weights, X[t_idx_, :]))
            label_probabilities = preds * y[t_idx_] + (1 - preds) * (1 - y[t_idx_])
            #print(label_probabilities)
            loglik = -np.sum(np.log(label_probabilities))

            num_unpack = 3
            reg = 0
            # reg_l1 = np.sum(np.abs(flattened)) * 1.
            for idx_ in range(0, len(weights), num_unpack):
                param_temp_ = weights[idx_:idx_+num_unpack]
                flattened, _ = weights_flatten(param_temp_[:2])
                reg_l1 = np.sum(np.abs(flattened)) * 1.
                reg += reg_l1
            return loglik + reg

        training_gradient_fun = grad(training_loss)
        param_ = adam(training_gradient_fun, self.partial_param_, callback=callback, step_size=step_size, num_iters=num_iters)

        self.partial_param_ = param_
        self.is_partial = True
        return self

    def partial_fit(self, X, y):
        check_is_fitted(self, 'base_model_')
        if self.is_partial:
            self.partial_fit_param(X, y)
        else:
            self.partial_fit_base(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, 'base_model_')
        if not self.is_partial:
            return self.base_model_.predict(X)
        else:
            multi_class = self.n_classes_ > 2
            model_ = gbm_gen(self.partial_param_, X, self.base_param_[2], self.base_param_[1], multi_class, self.n_classes_)
            preds = model_(self.partial_param_, X)
            if self.n_classes_ > 2:
                return np.argmax(preds, axis=1)
            else:
                return np.round(sigmoid(preds))


if __name__ == "__main__":
    # these are test cases - to be refactored out.
    from sklearn.datasets import make_classification
    X, y = make_classification(100, n_classes=3, n_informative=3, n_redundant=0, n_clusters_per_class=2, n_features=10)
    model = TGDClassifier(autograd_config={'num_iters': 5})
    model.fit(X, y)
    print(model.predict(X))

    # partial fit off lightgbm
    model.partial_fit(X, y)
    print(model.predict(X))

    # partial fit off itself
    model.partial_fit(X, y)
    print(model.predict(X))

    # test class binary
    X, y = make_classification(100, n_classes=2, n_informative=3, n_redundant=0, n_clusters_per_class=2, n_features=8)
    model = TGDClassifier(autograd_config={'num_iters': 100})
    model.fit(X, y)
    print(model.predict(X))
    print(np.round(model.predict_proba(X)))

    # partial fit off lightgbm
    model.partial_fit(X, y)
    print(model.predict(X))
    print(np.round(model.predict_proba(X)))

    # partial fit off itself
    model.partial_fit(X, y)
    print(model.predict(X))