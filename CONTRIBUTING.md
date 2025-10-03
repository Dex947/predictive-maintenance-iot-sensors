# Contributing to Predictive Maintenance for IoT Sensors

Thank you for your interest in contributing! This project aims to provide a robust, production-ready predictive maintenance system for IoT sensor data.

## 🎯 How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. **Use the issue template** (if available)
3. **Provide details**:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Error messages and stack traces

### Suggesting Enhancements

1. **Open an issue** with the `enhancement` label
2. **Describe the feature**:
   - Use case and motivation
   - Proposed implementation
   - Potential impact on existing code
   - Any breaking changes

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Follow the code style** (see below)
5. **Add tests** for new functionality
6. **Update documentation**
7. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: brief description"
   ```
8. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Code Style Guidelines

### Python Code Standards

- **PEP 8** compliance
- **Type hints** for function parameters and returns
- **Docstrings** for all functions, classes, and modules (Google style)
- **Maximum line length**: 100 characters
- **Imports**: Organized (standard library, third-party, local)

Example:
```python
from typing import List, Tuple

def process_sensor_data(
    data: pd.DataFrame,
    sensor_cols: List[str],
    window_size: int = 30
) -> Tuple[pd.DataFrame, int]:
    """
    Process sensor data with rolling window features.

    Args:
        data: Input dataframe with sensor readings
        sensor_cols: List of sensor column names
        window_size: Size of rolling window (default: 30)

    Returns:
        Tuple of (processed dataframe, number of features created)

    Raises:
        ValueError: If window_size < 1
    """
    # Implementation here
    pass
```

### Code Organization

- **Modular design**: One responsibility per function/class
- **Configuration-driven**: No hardcoded values
- **Logging**: Use logging module, not print statements
- **Error handling**: Proper try/except with informative messages
- **Type safety**: Use type hints and validate inputs

### Testing Requirements

All new code must include tests:

```python
# tests/test_your_module.py
import pytest
from src.your_module import your_function

def test_your_function_basic_case():
    """Test basic functionality."""
    result = your_function(input_data)
    assert result == expected_output

def test_your_function_edge_case():
    """Test edge cases."""
    with pytest.raises(ValueError):
        your_function(invalid_input)
```

Run tests:
```bash
pytest tests/ -v
```

## 🔧 Development Setup

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/predictive-maintenance-iot.git
cd predictive-maintenance-iot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Development Dependencies

```bash
pip install pytest pytest-cov black flake8 mypy
```

### 5. Run Tests

```bash
pytest tests/ -v --cov=src
```

## 🎨 Code Formatting

### Use Black for formatting:
```bash
black src/ tests/
```

### Use flake8 for linting:
```bash
flake8 src/ tests/ --max-line-length=100
```

### Use mypy for type checking:
```bash
mypy src/
```

## 📚 Documentation

### Docstring Format (Google Style)

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of function.

    Longer description if needed, explaining the purpose and behavior.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not a string

    Examples:
        >>> example_function(5, "test")
        True
    """
    pass
```

### Update Documentation

- Update `README.md` for user-facing changes
- Update docstrings for code changes
- Add examples for new features
- Update configuration guide if needed

## 🧪 Testing Guidelines

### Test Coverage Requirements

- **Minimum coverage**: 80%
- **Critical paths**: 100% coverage
- **Edge cases**: Must be tested

### Test Categories

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test module interactions
3. **End-to-End Tests**: Test full pipeline

### Example Test Structure

```python
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import FeatureEngineer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'unit_id': [1] * 100,
        'cycle': range(1, 101),
        'sensor_1': np.random.randn(100)
    })

@pytest.fixture
def config():
    """Load test configuration."""
    return {
        'preprocessing': {
            'window_size': 30,
            'statistical_features': ['mean', 'std']
        }
    }

def test_feature_engineer_creates_features(sample_data, config):
    """Test that feature engineering creates expected features."""
    engineer = FeatureEngineer(config)
    result = engineer.create_rolling_features(sample_data, ['sensor_1'])

    assert 'sensor_1_rolling_mean' in result.columns
    assert 'sensor_1_rolling_std' in result.columns
    assert len(result) == len(sample_data)

def test_feature_engineer_handles_empty_data(config):
    """Test feature engineer with empty dataframe."""
    engineer = FeatureEngineer(config)
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError):
        engineer.create_rolling_features(empty_df, ['sensor_1'])
```

## 🐛 Bug Fixes

### For Bug Fixes:

1. **Write a failing test** that reproduces the bug
2. **Fix the bug**
3. **Verify the test passes**
4. **Add regression test** to prevent recurrence

### Bug Fix Checklist:

- [ ] Issue number referenced in PR
- [ ] Failing test added
- [ ] Bug fixed
- [ ] Test passes
- [ ] Documentation updated if needed
- [ ] No breaking changes (or clearly marked)

## ✨ Feature Development

### Before Starting:

1. **Open an issue** to discuss the feature
2. **Get feedback** from maintainers
3. **Create a design document** for large features

### Development Checklist:

- [ ] Feature documented in issue
- [ ] Code follows style guidelines
- [ ] Tests added (unit + integration)
- [ ] Docstrings updated
- [ ] README updated if needed
- [ ] Configuration updated if needed
- [ ] No breaking changes (or versioned properly)

## 📋 Commit Message Guidelines

### Format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types:

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting)
- **refactor**: Code refactoring
- **test**: Adding/updating tests
- **chore**: Maintenance tasks

### Examples:

```
feat(preprocessing): add polynomial features

Added polynomial feature generation for capturing non-linear
relationships between sensors. Includes degree parameter in config.

Closes #123
```

```
fix(data_loader): handle missing RUL file gracefully

Previously crashed when RUL file was missing. Now raises
informative ValueError with suggestion to download dataset.

Fixes #456
```

## 🔍 Code Review Process

### For Contributors:

1. **Self-review** your code first
2. **Run all tests** locally
3. **Check code coverage**
4. **Update documentation**
5. **Request review** when ready

### For Reviewers:

- **Be constructive** and respectful
- **Explain reasoning** for requested changes
- **Approve** when ready or **request changes** with clear feedback

## 🎯 Priority Areas for Contribution

### High Priority:

1. **Deep Learning Models**: LSTM, GRU, CNN implementations
2. **Additional Datasets**: Support for PHM Society datasets
3. **Real-time Streaming**: Kafka/MQTT integration
4. **API Development**: Flask/FastAPI REST API
5. **Hyperparameter Tuning**: Optuna/Ray Tune integration

### Medium Priority:

6. **Cross-validation**: K-fold for robust evaluation
7. **Model Ensembles**: Stacking/blending methods
8. **Feature Selection**: Automated feature selection
9. **Visualization**: Interactive dashboards
10. **Documentation**: Tutorials and examples

### Low Priority:

11. **Performance Optimization**: Cython/numba acceleration
12. **Docker Support**: Containerization
13. **CI/CD Pipeline**: GitHub Actions
14. **Monitoring**: MLflow integration
15. **Database Support**: TimescaleDB integration

## 📞 Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Email**: For private concerns

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Acknowledgments

We appreciate all contributions, big or small! Every contribution helps improve the project.

### Contributors

Thank you to all contributors who have helped improve this project!

---

**Happy Contributing!** 🚀

For questions, open an issue or start a discussion.
