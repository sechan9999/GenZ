# Testing Guide for Gen Z Agent

**Version**: 2.0.0
**Last Updated**: 2025-11-20

## Overview

This guide covers the comprehensive testing infrastructure for Gen Z Agent, including unit tests, integration tests, and CI/CD pipelines.

## Test Structure

```
tests/
├── test_gen_z_agent/          # Gen Z Agent tests
│   ├── conftest.py            # Fixtures and configuration
│   ├── test_config.py         # Configuration tests
│   ├── test_agents/           # Agent tests
│   │   └── test_extractor.py
│   └── test_tools/            # Tool tests
│       ├── test_pdf_parser.py
│       ├── test_html_parser.py
│       ├── test_excel_generator.py
│       └── test_markdown_generator.py
└── test_nvdrs/                # NVDRS pipeline tests
    ├── test_models.py
    └── test_pii_redaction.py
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=gen_z_agent --cov-report=html
```

### Run Specific Test Suites

```bash
# Gen Z Agent tests only
pytest tests/test_gen_z_agent

# NVDRS tests only
pytest tests/test_nvdrs

# Specific test file
pytest tests/test_gen_z_agent/test_config.py

# Specific test function
pytest tests/test_gen_z_agent/test_config.py::test_config_candidates
```

### Run Tests by Marker

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Agent tests
pytest -m agents

# Tool tests
pytest -m tools

# Tests requiring API keys (skip in CI)
pytest -m "not requires_api"

# Korean language tests
pytest -m korean
```

## Test Categories

### 1. Unit Tests

Test individual functions and methods in isolation.

**Example:**
```python
def test_normalize_korean_text():
    """Test Korean text normalization."""
    text = "이재명   득표수"
    normalized = KoreanPDFParser.normalize_korean_text(text)
    assert normalized == "이재명 득표수"
```

**Run:**
```bash
pytest -m unit
```

### 2. Integration Tests

Test interactions between components.

**Example:**
```python
def test_pdf_parser_integration(sample_pdf_path):
    """Test full PDF parsing workflow."""
    result = KoreanPDFParser.parse_pdf(sample_pdf_path)
    assert "text" in result
    assert "tables" in result
```

**Run:**
```bash
pytest -m integration
```

### 3. Agent Tests

Test CrewAI agent creation and configuration.

**Location:** `tests/test_gen_z_agent/test_agents/`

**Run:**
```bash
pytest -m agents
# or
pytest tests/test_gen_z_agent/test_agents/
```

### 4. Tool Tests

Test custom tools (PDF parser, Excel generator, etc.).

**Location:** `tests/test_gen_z_agent/test_tools/`

**Run:**
```bash
pytest -m tools
# or
pytest tests/test_gen_z_agent/test_tools/
```

## Fixtures

### Common Fixtures (conftest.py)

```python
@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""

@pytest.fixture
def sample_pdf_path(temp_dir):
    """Mock PDF file path."""

@pytest.fixture
def sample_html_path(temp_dir):
    """Sample HTML file with Korean election data."""

@pytest.fixture
def sample_election_data():
    """Sample election data structure."""

@pytest.fixture
def sample_analysis_data():
    """Sample analysis results."""

@pytest.fixture
def mock_llm():
    """Mock LLM for testing without API calls."""
```

### Using Fixtures

```python
def test_excel_generation(temp_dir, sample_analysis_data):
    """Test Excel report generation."""
    output_path = temp_dir / "report.xlsx"

    ExcelReportGenerator.generate_report(
        output_path=str(output_path),
        summary=sample_analysis_data
    )

    assert output_path.exists()
```

## Coverage

### Viewing Coverage Reports

```bash
# Generate coverage report
pytest --cov=gen_z_agent --cov-report=html

# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Coverage Targets

| Component | Target | Current |
|-----------|--------|---------|
| Tools | 80%+ | TBD |
| Agents | 70%+ | TBD |
| Config | 90%+ | TBD |
| Overall | 75%+ | TBD |

### Excluding from Coverage

Use `# pragma: no cover` for code that shouldn't be tested:

```python
if __name__ == "__main__":  # pragma: no cover
    main()
```

## Mocking

### Mocking API Calls

```python
def test_agent_without_api_call(mock_llm):
    """Test agent creation without calling API."""
    agent = create_extractor_agent(mock_llm)
    assert agent.llm.model == "claude-sonnet-4-5-20250929"
```

### Mocking File Operations

```python
def test_pdf_parsing_error(monkeypatch):
    """Test PDF parsing error handling."""
    def mock_open(*args, **kwargs):
        raise IOError("File not found")

    monkeypatch.setattr("builtins.open", mock_open)

    with pytest.raises(IOError):
        KoreanPDFParser.parse_pdf("fake.pdf")
```

## CI/CD Integration

### GitHub Actions Workflows

**1. Main Tests** (`.github/workflows/tests.yml`)
- Runs on: Push to main, develop, claude/* branches
- Python versions: 3.8, 3.9, 3.10, 3.11
- Steps: Lint, format check, test, coverage

**2. Code Quality** (`.github/workflows/code-quality.yml`)
- Runs: Black, isort, Flake8, Pylint, mypy
- Enforces code standards

**3. NVDRS Tests** (`.github/workflows/nvdrs-tests.yml`)
- Runs NVDRS pipeline tests separately
- Specific coverage for NVDRS module

### Running CI Checks Locally

```bash
# Simulate CI environment
export ANTHROPIC_API_KEY="test_key_for_ci"

# Run all checks
black --check gen_z_agent tests
isort --check gen_z_agent tests
flake8 gen_z_agent tests
pytest --cov=gen_z_agent --cov=nvdrs_pipeline
```

## Writing New Tests

### Test Template

```python
"""
Tests for [Component Name]
"""

import pytest
from gen_z_agent.[module] import [Class/Function]


class Test[ComponentName]:
    """Test [component description]."""

    def test_[specific_behavior](self, fixture_name):
        """Test that [component] [does something]."""
        # Arrange
        input_data = ...

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result == expected_output


    def test_[error_condition](self):
        """Test error handling for [scenario]."""
        with pytest.raises(ExpectedException):
            function_that_should_fail()
```

### Best Practices

1. **One assertion per test** (when possible)
2. **Descriptive test names** - `test_normalize_korean_text_removes_extra_spaces`
3. **AAA pattern** - Arrange, Act, Assert
4. **Use fixtures** - Don't repeat setup code
5. **Test edge cases** - Empty input, None, invalid data
6. **Test error paths** - Exceptions, validation failures
7. **Avoid flaky tests** - No random data, no network calls
8. **Fast tests** - Mock slow operations

### Example: Comprehensive Test

```python
class TestKoreanPDFParser:
    """Comprehensive tests for Korean PDF Parser."""

    def test_parse_valid_pdf(self, sample_pdf_path):
        """Test parsing valid PDF file."""
        result = KoreanPDFParser.parse_pdf(sample_pdf_path)
        assert "text" in result
        assert result["page_count"] > 0

    def test_parse_nonexistent_file(self):
        """Test parsing non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            KoreanPDFParser.parse_pdf("/fake/path.pdf")

    def test_normalize_korean_text_nfc(self):
        """Test Korean text is normalized to NFC form."""
        text = "한글"  # Could be NFD or NFC
        normalized = KoreanPDFParser.normalize_korean_text(text)
        # Check it's normalized (this is simplified)
        assert normalized == "한글"

    @pytest.mark.parametrize("name,expected", [
        ("이재명", True),
        ("김문수", True),
        ("홍길동", True),
        ("John", False),
        ("이", False),
        ("", False),
    ])
    def test_is_korean_name_parametrized(self, name, expected):
        """Test Korean name detection with multiple inputs."""
        assert KoreanPDFParser.is_korean_name(name) == expected
```

## Debugging Tests

### Verbose Output

```bash
# Show all test output
pytest -v -s

# Show specific test
pytest -v -s tests/test_gen_z_agent/test_config.py::test_config_candidates
```

### Debug with pdb

```python
def test_something():
    """Test with debugger."""
    import pdb; pdb.set_trace()
    result = function_under_test()
    assert result == expected
```

Run with:
```bash
pytest -s  # -s allows pdb to work
```

### Using pytest-pdb

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger on first failure
pytest -x --pdb
```

## Performance Testing

### Measure Test Duration

```bash
# Show slowest tests
pytest --durations=10

# Show all durations
pytest --durations=0
```

### Mark Slow Tests

```python
@pytest.mark.slow
def test_large_pdf_parsing():
    """This test is slow."""
    ...
```

Skip slow tests:
```bash
pytest -m "not slow"
```

## Troubleshooting

### Tests Failing Locally

**Issue:** "ModuleNotFoundError"

**Solution:**
```bash
# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/GenZ"
```

### Tests Pass Locally, Fail in CI

**Possible causes:**
1. Missing dependencies in CI
2. Environment variables not set
3. File paths are absolute (should be relative)
4. Tests depend on local files

**Solution:**
- Check `.github/workflows/tests.yml`
- Use fixtures for file paths
- Mock external dependencies

### Coverage Too Low

**Strategies:**
1. Add tests for uncovered lines
2. Remove dead code
3. Add `# pragma: no cover` for unreachable code
4. Test error paths

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Questions?** See `CLAUDE.md` or check existing tests for examples.
