```markdown
# Brazil-Stock-Bond-Correlation-Puzzle Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill covers the development patterns and conventions used in the `Brazil-Stock-Bond-Correlation-Puzzle` repository. The project is written in Python and explores the correlation between Brazilian stock and bond markets. This guide documents the coding style, file organization, and workflows to help contributors maintain consistency and efficiency.

## Coding Conventions

### File Naming
- Use **camelCase** for file names.
  - Example: `dataLoader.py`, `correlationAnalysis.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .dataLoader import loadData
    ```

### Export Style
- Use **named exports** (explicitly define what is exported).
  - Example:
    ```python
    def analyzeCorrelation(data):
        # function body

    __all__ = ['analyzeCorrelation']
    ```

### Commit Messages
- Freeform commit messages, no strict prefix.
- Average length: ~57 characters.
  - Example: `Added function to calculate rolling correlation`

## Workflows

### Data Analysis Workflow
**Trigger:** When you want to analyze the correlation between stocks and bonds.
**Command:** `/analyze-correlation`

1. Prepare your data files in the expected format.
2. Run the main analysis script (e.g., `correlationAnalysis.py`).
3. Review the output plots and statistics.

### Adding a New Analysis Module
**Trigger:** When you need to add a new analysis feature or module.
**Command:** `/add-analysis-module`

1. Create a new Python file using camelCase (e.g., `newFeature.py`).
2. Use relative imports to access shared utilities.
3. Define functions and add them to `__all__` for named exports.
4. Write or update tests in a corresponding `*.test.*` file.

### Running Tests
**Trigger:** When you want to verify code correctness.
**Command:** `/run-tests`

1. Identify test files matching the `*.test.*` pattern.
2. Run tests using your preferred Python test runner (e.g., `pytest`, `unittest`).
3. Review test results and fix any failures.

## Testing Patterns

- Test files follow the `*.test.*` naming pattern (e.g., `dataLoader.test.py`).
- Testing framework is not explicitly specified; use standard Python testing tools.
- Tests are typically placed alongside the modules they test.
- Example test structure:
  ```python
  import unittest
  from .dataLoader import loadData

  class TestDataLoader(unittest.TestCase):
      def test_load_data(self):
          data = loadData('test.csv')
          self.assertIsNotNone(data)
  ```

## Commands
| Command                | Purpose                                         |
|------------------------|-------------------------------------------------|
| /analyze-correlation   | Run the main stock-bond correlation analysis    |
| /add-analysis-module   | Scaffold a new analysis module                  |
| /run-tests             | Execute all tests in the repository             |
```