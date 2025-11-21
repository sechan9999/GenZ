# Migration Guide: Monolithic to Modular Architecture

**Version 1.0 → Version 2.0**
**Date**: 2025-11-20

## Overview

This guide helps you migrate from the original monolithic `main.py` (v1.0) to the new modular architecture (v2.0).

## What Changed?

### Before (v1.0) - Monolithic Architecture

```
gen_z_agent/
├── main.py (313 lines - all agents, tasks, crew in one file)
├── config.py
└── requirements.txt
```

### After (v2.0) - Modular Architecture

```
gen_z_agent/
├── agents/              # 5 agent modules
│   ├── extractor.py
│   ├── validator.py
│   ├── analyst.py
│   ├── reporter.py
│   └── communicator.py
├── tools/               # Custom tools
│   ├── pdf_parser.py
│   ├── html_parser.py
│   ├── excel_generator.py
│   └── markdown_generator.py
├── tasks/               # Task definitions
│   └── electoral_tasks.py
├── crew.py              # Crew configuration
├── main_refactored.py   # New entry point
├── config.py            # (unchanged)
└── requirements_updated.txt
```

## Breaking Changes

### 1. Entry Point Changed

**Old way:**
```bash
python gen_z_agent/main.py data/election.pdf
```

**New way:**
```bash
python -m gen_z_agent.main_refactored data/election.pdf
```

### 2. Import Changes

**Old way (internal imports):**
```python
# main.py had everything in one file
# No imports needed
```

**New way:**
```python
from gen_z_agent.agents import create_extractor_agent
from gen_z_agent.tools import PDFParserTool
from gen_z_agent.crew import run_electoral_analysis
```

### 3. Function Signatures

**Agent creation functions now require LLM parameter:**

**Old way:**
```python
# Agents were created inline with llm
extractor = Agent(
    role="...",
    llm=llm,
    ...
)
```

**New way:**
```python
# Agents created via factory functions
from gen_z_agent.agents import create_extractor_agent
extractor = create_extractor_agent(llm)
```

## Migration Steps

### Step 1: Update Dependencies

```bash
# Install updated dependencies
pip install -r gen_z_agent/requirements_updated.txt
pip install -r requirements-dev.txt
```

### Step 2: Update Your Code

If you were importing from `main.py`, update imports:

**Before:**
```python
from gen_z_agent.main import run_invoice_analysis
```

**After:**
```python
from gen_z_agent.crew import run_electoral_analysis
```

### Step 3: Update CLI Usage

**Before:**
```bash
python gen_z_agent/main.py \
    --file data/election.pdf \
    --id ELEC001 \
    --recipients "admin@example.com"
```

**After:**
```bash
python -m gen_z_agent.main_refactored \
    data/election.pdf \
    --id ELEC001 \
    --recipients "admin@example.com"
```

Note: Positional argument for file instead of `--file` flag.

### Step 4: Update Custom Agents (if any)

If you created custom agents based on the old structure:

**Before:**
```python
my_agent = Agent(
    role="Custom Agent",
    goal="...",
    backstory="...",
    tools=[file_reader],
    llm=llm,
    verbose=True
)
```

**After:**
Create a factory function in `gen_z_agent/agents/`:

```python
# gen_z_agent/agents/my_custom_agent.py
def create_my_custom_agent(llm, tools=None):
    if tools is None:
        tools = [FileReadTool()]

    return Agent(
        role="Custom Agent",
        goal="...",
        backstory="...",
        tools=tools,
        llm=llm,
        verbose=True,
        memory=True
    )
```

### Step 5: Run Tests

Verify everything works:

```bash
# Run tests
pytest tests/test_gen_z_agent -v

# Check code quality
black --check gen_z_agent tests
flake8 gen_z_agent tests
```

## New Features in v2.0

### 1. Custom PDF/HTML Parsers

You can now use dedicated parsers for Korean text:

```python
from gen_z_agent.tools import KoreanPDFParser

result = KoreanPDFParser.parse_pdf("election.pdf", extract_tables=True)
print(result["election_data"]["candidates"])
```

### 2. Enhanced Report Generation

Generate actual Excel and Markdown files (not just templates):

```python
from gen_z_agent.tools import ExcelReportGenerator, MarkdownReportGenerator

# Generate Excel
ExcelReportGenerator.generate_report(
    output_path="output/report.xlsx",
    raw_data=[...],
    analysis={...}
)

# Generate Markdown
MarkdownReportGenerator.generate_report(
    output_path="output/report.md",
    title="Election Analysis",
    candidates=[...]
)
```

### 3. Comprehensive Testing

v2.0 includes a full test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m agents
pytest -m tools
pytest -m integration
```

### 4. CI/CD Pipeline

Automated testing and code quality checks via GitHub Actions:

- `.github/workflows/tests.yml` - Main test suite
- `.github/workflows/code-quality.yml` - Linting and formatting
- `.github/workflows/nvdrs-tests.yml` - NVDRS pipeline tests

## Compatibility

### Backward Compatibility

The **old `main.py` still works** and will continue to work. You can use both versions side by side:

```bash
# Old version
python gen_z_agent/main.py data/election.pdf

# New version
python -m gen_z_agent.main_refactored data/election.pdf
```

### When to Migrate?

Migrate to v2.0 if you need:
- ✅ Better code organization and maintainability
- ✅ Custom PDF/HTML parsing for Korean text
- ✅ Actual file generation (Excel, Markdown)
- ✅ Comprehensive testing
- ✅ CI/CD integration
- ✅ Future-proof architecture

Stay on v1.0 if:
- ⚠️ You have existing integrations that depend on `main.py`
- ⚠️ You need stability and don't need new features
- ⚠️ Migration effort is not justified yet

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Ensure you're running from the project root:

```bash
cd /path/to/GenZ
python -m gen_z_agent.main_refactored data/election.pdf
```

### Issue: "No module named 'pdfplumber'"

**Solution**: Install updated dependencies:

```bash
pip install -r gen_z_agent/requirements_updated.txt
```

### Issue: Tests failing

**Solution**: Set up test environment:

```bash
pip install -r requirements-dev.txt
export ANTHROPIC_API_KEY="your_key_here"
pytest
```

### Issue: Import errors in agents

**Solution**: Check your Python path includes the project root:

```python
import sys
sys.path.insert(0, '/path/to/GenZ')
```

## Getting Help

- **Documentation**: See `CLAUDE.md` for architecture details
- **Examples**: Check `examples/` directory
- **Tests**: Look at `tests/test_gen_z_agent/` for usage examples
- **Issues**: Report problems on GitHub

## Rollback Plan

If you need to rollback to v1.0:

```bash
# Simply use the old entry point
python gen_z_agent/main.py data/election.pdf

# Or restore from git
git checkout <previous_commit> gen_z_agent/main.py
```

The old `main.py` is preserved and fully functional.

---

**Questions?** See `CLAUDE.md` or contact the development team.
