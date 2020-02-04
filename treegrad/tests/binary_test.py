from sklearn.datasets import make_classification
import numpy as np
from treegrad import TGDClassifier


def test_binary():
    # test class binary
    X, y = make_classification(
        100,
        n_classes=2,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=2,
        n_features=10,
    )
    model = TGDClassifier(autograd_config={"num_iters": 1})
    model.fit(X, y)
    assert model.predict(X).shape[0] == X.shape[0]
    a1 = model.predict_proba(X)
    assert a1.shape[1] == 2

    # partial fit off lightgbm
    model.partial_fit(X, y)
    assert model.predict(X).shape[0] == X.shape[0]
    a2 = model.predict_proba(X)
    assert a2.shape[1] == 2

    # partial fit off itself
    model.partial_fit(X, y)
    assert model.predict(X).shape[0] == X.shape[0]
    a3 = model.predict_proba(X)
    assert a3.shape[1] == 2

    assert not np.array_equal(a1, a2)
    assert not np.array_equal(a1, a3)
