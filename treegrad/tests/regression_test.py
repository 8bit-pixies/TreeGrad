from sklearn.datasets import make_regression
import numpy as np
from treegrad import TGDRegressor
from sklearn.metrics import mean_squared_error




def test_regressor():
    # test class binary
    X, y = make_regression()
    model = TGDRegressor(autograd_config={"num_iters": 1})
    model.fit(X, y)
    a1 = model.predict(X)
    assert a1.shape[0] == X.shape[0]

    # partial fit off lightgbm
    model.partial_fit(X, y)
    a2 = model.predict(X)
    assert a2.shape[0] == X.shape[0]

    # partial fit off itself
    model.partial_fit(X, y)
    a3 = model.predict(X)
    assert a3.shape[0] == X.shape[0]

    err1 = mean_squared_error(y, a1)
    err2 = mean_squared_error(y, a2)
    err3 = mean_squared_error(y, a3)

    print(err1)
    print(err2)
    print(err3)

    assert not np.array_equal(a1, a2)
    assert not np.array_equal(a1, a3)
