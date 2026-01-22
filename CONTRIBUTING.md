# Contributing to Geometric Safety Features

Thank you for your interest in contributing to geometric safety features! This document provides guidelines for contributors.

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Maintain scientific integrity in discussions

## How to Contribute

### Reporting Issues
- Use GitHub Issues for bug reports and feature requests
- Include:
  - Clear description of the issue
  - Steps to reproduce
  - Expected vs. actual behavior
  - System information (Python version, OS)

### Development Workflow

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/yourusername/geometric_safety_features.git`
3. **Create a branch** for your changes: `git checkout -b feature/your-feature-name`
4. **Install** in development mode: `pip install -e ".[dev]"`
5. **Make changes** following the coding standards
6. **Run tests**: `pytest tests/`
7. **Commit** with clear messages: `git commit -m "feat: add new feature"`
8. **Push** to your fork: `git push origin feature/your-feature-name`
9. **Create a Pull Request** with a clear description

### Coding Standards

- **Python**: Follow PEP 8
- **Imports**: Use absolute imports, group by standard library, third-party, local
- **Documentation**: Add docstrings to all public functions (Google/NumPy style)
- **Testing**: Write tests for new features, maintain >85% coverage
- **Type Hints**: Use type annotations for better IDE support

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=mirrorfield --cov-report=html

# Run specific test
pytest tests/test_geometry.py::TestGeometryBundle::test_bundle_compute
```

### Documentation

- Update docstrings for any API changes
- Add examples for new features in `examples/`
- Update README.md if adding new functionality

### Commit Messages

Follow conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for test additions
- `refactor:` for code improvements

### Areas for Contribution

**High Priority:**
- Performance optimizations for large-scale embeddings
- Additional baseline methods for comparison
- Integration with popular ML frameworks

**Research:**
- Novel geometric features for uncertainty detection
- Theoretical analysis of feature effectiveness
- Applications to new domains (vision, multimodal)

**Infrastructure:**
- CI/CD improvements
- Documentation enhancements
- Package distribution improvements

## Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Email**: dillanjc91@gmail.com for private matters

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for helping advance AI safety through geometric analysis! 🔒🤖