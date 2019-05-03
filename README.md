# TreeGrad

`TreeGrad` implements a naive approach to converting a Gradient Boosted Tree Model to an Online trainable model. It does this by creating differentiable tree models which can be learned via auto-differentiable frameworks. `TreeGrad` is in essence an implementation of Kontschieder, Peter, et al. "Deep neural decision forests." with extensions.

To install

```
python setup.py install
```

or alternatively from pypi


```
pip install treegrad
```

Run tests:

```
python -m nose2
```

```
@article{siu2019treegrad,
  title={TreeGrad: Transferring Tree Ensembles to Neural Networks},
  author={Siu, Chapman},
  journal={arXiv preprint 1904.11132},
  year={2019}
}
```


# Usage

```py
from sklearn.
import treegrad as tgd

mod = tgd.TGDClassifier(num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, autograd_config={'refit_splits':False})
mod.fit(X, y)
mod.partial_fit(X, y)
```

# Requirments

The requirements for this package are:

*  lightgbm
*  scikit-learn
*  autograd

Future plans:

*  Add implementation for Neural Architecture search for decision boundary splits (requires a bit of clean up - TBA)
   *  Implementation can be done quite trivially using objects residing in `tree_utils.py` - Challenge is getting this working in a sane manner with `scikit-learn` interface.
*  GPU enabled auto differentiation framework - see `notebooks/` for progress off Colab for Tensorflow 2.0 port
*  support xgboost/lightgbm additional features such as monotone constraints
*  Support `RegressorMixin`

# Results

When decision splits are reset and subsequently re-learned, TreeGrad can be competitive in performance with popular implementations (albeit an order of magnitude slower). Below is a table showing accuracy on test dataset on UCI benchmark datasets for Boosted Ensemble models (100 trees)


| Dataset  | TreeGrad  | LightGBM  | Scikit-Learn (Gradient Boosting Classifier) |
| ---------| --------- | --------- | ------------------------------------------- |
| adult    | 0.860     | 0.873     | **0.874**                                   |
| covtype  | 0.832     | **0.835** | 0.826                                       |
| dna      | **0.950** | 0.949     | 0.946                                       |
| glass    | 0.766     | **0.813** | 0.719                                       |
| mandelon | **0.882** | 0.881     | 0.866                                       |
| soybean  | **0.936** | **0.936** | 0.917                                       |
| yeast    | **0.591** | 0.573     | 0.542                                       |


# Implementation

<!-- insert link to arxiv paper -->

To understand the implementation of `TreeGrad`, we interpret a decision tree algorithm to be a three layer neural network, where the layers are as follows:

1.  Node layer, which determines the decision boundaries
2.  Routing layer, which determines which nodes are used to route to the final leaf nodes
3.  Leaf layer, the layer which determines the final predictions

In the node layer, the decision boundaries can be interpreted as _axis-parallel_ decision boundaries from your typical Linear Classifier; i.e. a fully connected dense layer

The routing layer requires a binary routing matrix to which essentially the global product routing is applied

The leaf layer is your typical fully connected dense layer.

This approach is the same as the one taken by Kontschieder, Peter, et al. "Deep neural decision forests."

