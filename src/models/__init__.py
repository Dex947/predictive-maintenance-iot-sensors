"""Model modules for predictive maintenance system."""

from .baseline_models import BaselineModel, EnsembleModel

# Optional deep learning import
try:
    from .deep_learning_models import DeepLearningModel
    __all__ = [
        'BaselineModel',
        'EnsembleModel',
        'DeepLearningModel',
    ]
except ImportError:
    DeepLearningModel = None
    __all__ = [
        'BaselineModel',
        'EnsembleModel',
    ]
