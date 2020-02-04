from treegrad.treegrad import TGDClassifier, TGDRegressor
import warnings

try:
    from treegrad.version import version as __version__  # NOQA
except:
    warnings.warn("Could not import version, has package been installed?")
