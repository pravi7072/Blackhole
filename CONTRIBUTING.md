# Contributing to BLACKHOLE AI

First off, thank you for considering contributing to BLACKHOLE AI! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples**
- **Describe the behavior you observed**
- **Explain which behavior you expected to see**
- **Include screenshots if applicable**
- **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List some examples of how it would be used**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the coding standards
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Commit your changes** with clear commit messages
6. **Push to your fork** and submit a pull request

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 120 characters
- Use meaningful variable and function names
- Add docstrings to all functions and classes

### Code Structure

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    # Implementation
    return True
```

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Reference issues and pull requests when relevant
- First line should be 50 characters or less
- Provide detailed description in commit body if needed

Example:
```
Add documentation search feature

- Implemented automatic detection of code queries
- Added official documentation scraping
- Integrated results into context builder
- Added tests for documentation search

Fixes #123
```

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Run self-tests: `python BlackholeFile.py --selftest`
- Test with different API configurations
- Test edge cases and error handling

### Documentation

- Update README.md for new features
- Add inline comments for complex logic
- Update docstrings when changing function behavior
- Create examples for new features

## Development Setup

1. **Clone your fork:**
   ```bash
   git clone https://github.com/pravi7072/Blackhole.git
   cd Blackhole
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Create feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Project Structure

```
Blackhole/
├── BlackholeFile.py  # Main application
├── requirements.txt                   # Dependencies
├── .env.example                      # Environment template
├── README.md                         # Documentation
├── CONTRIBUTING.md                   # This file
├── LICENSE                           # MIT License
└── .gitignore                        # Git ignore rules
```

## Code Review Process

1. **Maintainer review** - Code will be reviewed by project maintainers
2. **Testing** - All tests must pass
3. **Documentation** - Documentation must be updated
4. **Approval** - At least one maintainer approval required
5. **Merge** - Once approved, changes will be merged

## Community

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Provide constructive feedback
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)

## Questions?

Feel free to ask questions in:
- [GitHub Discussions](https://github.com/yourusername/blackhole-ai/discussions)
- [GitHub Issues](https://github.com/yourusername/blackhole-ai/issues)

Thank you for contributing! 🚀
