# CLAUDE.md - AI Assistant Guide for Gen Z Agent Project

## 📋 Project Overview

**Gen Z Agent** is a multi-agent system using CrewAI + Anthropic Claude for automated analysis of Korean election data and invoices. The system coordinates 5 specialized AI agents to extract, validate, analyze, and report on electoral data from PDF/HTML documents.

**Current Status**: ✅ Core Implementation Complete - The multi-agent system is functional with all 5 agents implemented in a monolithic architecture. Ready for production use with ongoing enhancements.

**Key Technologies**:
- CrewAI (Multi-agent orchestration)
- Anthropic Claude (claude-sonnet-4-5-20250929)
- Python 3.8+
- PDF/HTML parsing (pdfplumber, PyPDF2, BeautifulSoup)
- Data processing (pandas, numpy)
- Visualization (matplotlib, plotly)
- Excel output (openpyxl, xlsxwriter)

## 🏗️ Repository Structure

### Current Structure (as of 2025-11-20)
```
GenZ/
├── .git/                      # Git repository
├── README.md                  # Project documentation (Korean)
├── CLAUDE.md                  # This file - AI assistant guide
├── env.example                # Environment variables template
├── gen_z_agent/               # Main application package
│   ├── main.py                # Complete 5-agent implementation (313 lines)
│   ├── config.py              # Configuration management (216 lines)
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example           # Environment template
│   ├── .gitignore             # Git ignore rules
│   ├── README.md              # User-facing documentation (Korean)
│   ├── GenZ.zip               # Packaged distribution
│   ├── invoices/              # Input documents directory
│   │   └── README.md
│   ├── output/                # Generated reports directory
│   │   └── README.md
│   ├── historical/            # Historical data for comparisons
│   │   └── README.md
│   └── utils/                 # Utility functions directory
│       └── README.md
├── docs/                      # Additional documentation
│   └── palantir_foundry_ehr_integration.md  # Foundry integration guide
└── examples/                  # Example configurations
    ├── README_FOUNDRY_EHR.md
    └── foundry_ehr_quickstart.yml
```

**Key Implementation Notes**:
- ✅ **All 5 agents implemented** in `gen_z_agent/main.py` (monolithic architecture)
- ✅ **Configuration management** complete in `gen_z_agent/config.py`
- ✅ **Environment handling** with dotenv support
- ✅ **CLI interface** with argparse
- ✅ **Korean election data support** with 5 candidate profiles
- ✅ **Report templates** for Excel and Markdown
- ⚠️ **Testing infrastructure** not yet implemented
- ⚠️ **Modular architecture** (separate agent files) planned but not yet refactored
- ⚠️ **CI/CD pipelines** not yet configured

### Target Modular Structure (Future Refactoring Goal)

The following structure represents the ideal modular architecture for improved maintainability and testability. This is a **future refactoring goal** - the current implementation works but could benefit from this organization:

```
GenZ/
├── .github/
│   └── workflows/      # CI/CD pipelines (TODO)
├── gen_z_agent/        # Main application package
│   ├── __init__.py
│   ├── agents/         # 5 specialized agents (REFACTOR: currently in main.py)
│   │   ├── __init__.py
│   │   ├── extractor.py       # Invoice Data Extractor
│   │   ├── validator.py       # Data Validator & Enricher
│   │   ├── analyst.py         # Electoral Data Analyst
│   │   ├── reporter.py        # Executive Report Writer
│   │   └── communicator.py    # Communication Agent
│   ├── tasks/          # CrewAI task definitions (TODO: extract from main.py)
│   │   ├── __init__.py
│   │   └── electoral_tasks.py
│   ├── tools/          # Custom tools for agents (TODO)
│   │   ├── __init__.py
│   │   ├── pdf_parser.py
│   │   ├── html_parser.py
│   │   ├── korean_ocr.py
│   │   └── data_enrichment.py
│   ├── models/         # Data models and schemas (TODO)
│   │   ├── __init__.py
│   │   ├── electoral_data.py
│   │   └── validation_rules.py
│   ├── utils/          # Utility functions (EXISTS: needs population)
│   │   ├── __init__.py
│   │   ├── logger.py           # TODO
│   │   ├── config.py           # TODO: move from gen_z_agent/config.py
│   │   └── file_handlers.py    # TODO
│   ├── output/         # Generated reports (EXISTS)
│   ├── invoices/       # Input documents (EXISTS)
│   ├── historical/     # Historical data (EXISTS)
│   ├── temp/           # Temporary files (TODO: not yet created)
│   ├── crew.py         # CrewAI crew configuration (TODO: extract from main.py)
│   ├── config.py       # Configuration (EXISTS ✅)
│   └── main.py         # Entry point (EXISTS ✅)
├── tests/              # Test suite (TODO)
│   ├── __init__.py
│   ├── test_agents/
│   ├── test_tools/
│   ├── test_integration/
│   └── fixtures/       # Test data
├── docs/               # Additional documentation (EXISTS: partially)
│   ├── palantir_foundry_ehr_integration.md  # EXISTS ✅
│   ├── architecture.md                      # TODO
│   ├── agent_design.md                      # TODO
│   └── api_reference.md                     # TODO
├── examples/           # Sample configurations (EXISTS: Foundry examples)
│   ├── README_FOUNDRY_EHR.md               # EXISTS ✅
│   └── foundry_ehr_quickstart.yml          # EXISTS ✅
├── scripts/            # Utility scripts (TODO)
│   ├── setup.sh
│   └── run_analysis.py
├── .env                # Local environment (gitignored)
├── .gitignore          # EXISTS ✅
├── requirements.txt    # Python dependencies (EXISTS ✅)
├── requirements-dev.txt # Development dependencies (TODO)
├── setup.py            # Package setup (TODO)
├── pytest.ini          # Pytest configuration (TODO)
├── CLAUDE.md          # This file (EXISTS ✅)
└── README.md          # User-facing documentation (EXISTS ✅)
```

**Refactoring Priority**: Low - Current monolithic implementation is functional and maintainable for the current scope. Consider refactoring when:
- Adding more than 2-3 additional agents
- Team size grows beyond 2 developers
- Need for independent agent testing increases
- Code complexity exceeds ~500 lines in main.py

## 🤖 The 5 Agent Architecture

### 1. Invoice Data Extractor (청구서 데이터 추출 전문가)
- **Role**: Extract structured data from PDF/HTML documents
- **Responsibilities**:
  - Read and parse Korean election count sheets (개표상황표)
  - Apply OCR when needed
  - Extract tables with candidate names, vote counts, regions
  - Handle multiple document formats
- **Output**: Raw structured data (JSON/dict)
- **File**: `gen_z_agent/main.py:34-45` (lines 34-45 in main.py)

### 2. Data Validator & Enricher (데이터 검증 및 보강 전문가)
- **Role**: Ensure data integrity and enrich with external sources
- **Responsibilities**:
  - Validate data completeness and correctness
  - Check sum calculations (vote counts match totals)
  - Detect missing or anomalous values
  - Enrich with external data (demographics, historical data)
- **Output**: Validated and enriched dataset
- **File**: `gen_z_agent/main.py:47-58` (lines 47-58 in main.py)

### 3. Electoral Data Analyst (선거 데이터 분석가)
- **Role**: Perform statistical analysis and pattern detection
- **Responsibilities**:
  - Calculate statistics (mean, median, std dev, percentiles)
  - Detect anomalous patterns (outliers, irregularities)
  - Identify trends and correlations
  - Generate insights and findings
- **Output**: Analysis results with statistical measures
- **File**: `gen_z_agent/main.py:60-70` (lines 60-70 in main.py)

### 4. Executive Report Writer (보고서 작성 전문가)
- **Role**: Create professional multi-format reports
- **Responsibilities**:
  - Generate Excel spreadsheets with formatted data
  - Create Markdown reports with visualizations
  - Produce PDF executive summaries
  - Include charts, tables, and key findings
- **Output**: Report files (Excel, Markdown, PDF)
- **File**: `gen_z_agent/main.py:72-83` (lines 72-83 in main.py)

### 5. Communication Agent (커뮤니케이션 담당자)
- **Role**: Notify stakeholders of results
- **Responsibilities**:
  - Send email notifications with attachments
  - Post Slack messages with summary
  - Handle communication errors gracefully
  - Format messages appropriately for each channel
- **Output**: Delivery confirmations
- **File**: `gen_z_agent/main.py:85-94` (lines 85-94 in main.py)

## 🔄 Agent Workflow

```
Input Document (PDF/HTML)
        ↓
[1. Extractor] → Raw Data
        ↓
[2. Validator] → Validated & Enriched Data
        ↓
[3. Analyst] → Analysis Results & Insights
        ↓
[4. Reporter] → Reports (Excel, MD, PDF)
        ↓
[5. Communicator] → Stakeholder Notifications
```

**Sequential Processing**: Each agent depends on the previous agent's output. Use CrewAI's sequential task execution.

## 🏥 Palantir Foundry Integration (Healthcare EHR Extension)

**Added**: 2025-11-20

This project includes comprehensive documentation for integrating with Palantir Foundry to process FHIR healthcare data through Azure Event Hubs and Databricks Delta Lake. This represents a parallel use case beyond election data analysis.

### Key Resources

- **📄 Comprehensive Guide**: `docs/palantir_foundry_ehr_integration.md` (1,248 lines)
  - Data connection configurations (Azure Event Hubs, ADLS Gen2)
  - Dataset definitions for Bronze/Silver/Gold layers
  - PySpark transformations for FHIR normalization
  - Pipeline orchestration and monitoring
  - Security and HIPAA compliance configurations
  - Foundry Ontology modeling for FHIR resources

- **📄 Quick Start**: `examples/README_FOUNDRY_EHR.md`
  - Getting started guide
  - Architecture overview
  - Prerequisites and setup steps

- **📄 Configuration**: `examples/foundry_ehr_quickstart.yml`
  - Minimal working configuration
  - Quick deployment template

### Architecture Overview

```
Azure Event Hubs (FHIR) → Databricks Delta Lake (Bronze/Silver/Gold)
                                    ↓
                          Palantir Foundry Analytics
                          - Clinical Dashboards
                          - ML/AI Models
                          - Healthcare Applications
```

### Integration Approaches

1. **Delta Lake Direct Connection** - Recommended for analytics on processed data
2. **Event Hubs Streaming** - For parallel real-time processing
3. **Databricks Lakehouse Federation** - For federated queries

### FHIR Resources Supported

- **Observation** - Lab results, vital signs, clinical observations
- **MedicationStatement** - Active and historical medications
- **Patient** - Demographics and identifiers (PHI-protected)
- **Encounter** - Clinical visits and episodes

### Compliance & Security

- HIPAA-compliant data governance
- PHI classification and encryption (AES-256)
- Audit logging with 7-year retention
- Role-based access control (RBAC)

### Use Cases

1. **Clinical Analytics** - Patient vital sign trends, medication adherence
2. **Population Health** - Cohort identification (e.g., hypertension patients)
3. **Data Quality Monitoring** - Real-time pipeline health checks
4. **Research Analytics** - De-identified data for healthcare research

**Note**: This is documentation for integration with an external EHR pipeline project. The Gen Z Agent core functionality focuses on Korean election data analysis.

## 🛠️ Development Guidelines

### Code Style
- **Python Version**: 3.8+ (use type hints)
- **Style Guide**: PEP 8
- **Formatter**: Black (line length: 100)
- **Linter**: Flake8 + pylint
- **Import Order**: isort
- **Type Checking**: mypy (optional but recommended)

### Naming Conventions
- **Files**: lowercase_with_underscores.py
- **Classes**: PascalCase (e.g., `InvoiceDataExtractor`)
- **Functions/Methods**: snake_case (e.g., `extract_election_data`)
- **Constants**: UPPER_CASE (e.g., `MAX_RETRIES`)
- **Private methods**: _leading_underscore (e.g., `_parse_table`)

### Documentation
- **Docstrings**: Google style
- **Module docstrings**: Explain purpose and main components
- **Class docstrings**: Describe responsibility and usage
- **Function docstrings**: Args, Returns, Raises, Examples
- **Comments**: Korean and English both acceptable (Korean for domain-specific terms)

Example:
```python
def extract_candidate_votes(pdf_path: str) -> Dict[str, int]:
    """Extract candidate names and vote counts from election PDF.

    선거 개표상황표 PDF에서 후보자명과 득표수를 추출합니다.

    Args:
        pdf_path: Path to the election count sheet PDF file

    Returns:
        Dictionary mapping candidate names to vote counts

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        PDFParseError: If PDF structure is invalid

    Example:
        >>> votes = extract_candidate_votes("election_2024.pdf")
        >>> print(votes)
        {'김철수': 15234, '이영희': 13891}
    """
```

### Error Handling
- **Use specific exceptions**: Create custom exception classes
- **Fail gracefully**: Log errors, provide context, suggest fixes
- **Retry logic**: For network operations (API calls, web scraping)
- **Validation**: Validate inputs at entry points
- **Logging**: Use Python's logging module, not print statements

Example:
```python
class ElectionDataError(Exception):
    """Base exception for election data processing."""
    pass

class PDFParseError(ElectionDataError):
    """Raised when PDF parsing fails."""
    pass

class ValidationError(ElectionDataError):
    """Raised when data validation fails."""
    pass
```

### Testing
- **Framework**: pytest
- **Coverage**: Aim for >80%
- **Test Types**:
  - Unit tests: Individual functions and methods
  - Integration tests: Agent interactions
  - End-to-end tests: Full workflow
- **Fixtures**: Use pytest fixtures for test data
- **Mocking**: Mock external APIs and file I/O

Test file naming: `test_<module_name>.py`

Example:
```python
# tests/test_agents/test_extractor.py
import pytest
from gen_z_agent.agents.extractor import InvoiceDataExtractor

@pytest.fixture
def sample_pdf_path():
    return "tests/fixtures/sample_election.pdf"

def test_extract_candidate_votes(sample_pdf_path):
    extractor = InvoiceDataExtractor()
    votes = extractor.extract_candidate_votes(sample_pdf_path)
    assert isinstance(votes, dict)
    assert len(votes) > 0
    assert all(isinstance(v, int) for v in votes.values())
```

### Logging
- **Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Format**: Timestamp, level, module, message
- **Configuration**: Load from environment variable `LOG_LEVEL`
- **Agent logs**: Each agent should log its inputs, outputs, and key decisions

Example:
```python
import logging

logger = logging.getLogger(__name__)

def extract_data(file_path: str):
    logger.info(f"Starting data extraction from {file_path}")
    try:
        data = _parse_file(file_path)
        logger.info(f"Successfully extracted {len(data)} records")
        return data
    except Exception as e:
        logger.error(f"Failed to extract data: {e}", exc_info=True)
        raise
```

### Configuration Management
- **Environment Variables**: Use python-dotenv to load .env file
- **Config Class**: Create a centralized config class
- **Validation**: Validate all required environment variables at startup
- **Defaults**: Provide sensible defaults where possible

Example:
```python
# gen_z_agent/utils/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY", None)

    # Claude Settings
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
    CLAUDE_TEMPERATURE = float(os.getenv("CLAUDE_TEMPERATURE", "0"))
    CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))

    # Directories
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./gen_z_agent/output")
    TEMP_DIR = os.getenv("TEMP_DIR", "./gen_z_agent/temp")

    # Analysis Settings
    ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "2.0"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required")
```

## 🔌 Dependencies

### Core Dependencies (requirements.txt)
```
crewai>=0.30.0
anthropic>=0.18.0
python-dotenv>=1.0.0
pydantic>=2.0.0

# Document Processing
pdfplumber>=0.10.0
PyPDF2>=3.0.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0
xlsxwriter>=3.1.0

# Visualization
matplotlib>=3.7.0
plotly>=5.14.0

# Communication
requests>=2.31.0

# Utilities
tqdm>=4.65.0
python-dateutil>=2.8.0
```

### Development Dependencies (requirements-dev.txt)
```
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
black>=23.7.0
flake8>=6.1.0
pylint>=2.17.0
isort>=5.12.0
mypy>=1.4.0
```

## 🎯 Implementation Status & Priorities

**Last Updated**: 2025-11-20

### ✅ COMPLETED PHASES

#### Phase 1: Foundation ✅ (COMPLETE)
1. ✅ Set up project structure and files
2. ✅ Create requirements.txt with dependencies
3. ✅ Implement configuration management (config.py - 216 lines)
4. ✅ Set up logging infrastructure (basic, in config.py)
5. ✅ Create base exception classes (basic error handling in place)
6. ✅ Write .gitignore file

#### Phase 4: Agents ✅ (COMPLETE - Monolithic)
1. ✅ InvoiceDataExtractor - Implemented in main.py:34-45
2. ✅ DataValidatorEnricher - Implemented in main.py:47-58
3. ✅ ElectoralDataAnalyst - Implemented in main.py:60-70
4. ✅ ExecutiveReportWriter - Implemented in main.py:72-83
5. ✅ CommunicationAgent - Implemented in main.py:85-94

All agents include:
- ✅ Agent configuration with role, goal, backstory
- ✅ LLM integration (Claude Sonnet 4.5)
- ✅ Basic tools (FileReadTool, SerperDevTool)
- ⚠️ Unit tests - NOT YET IMPLEMENTED

#### Phase 5: CrewAI Integration ✅ (COMPLETE)
1. ✅ Define tasks in main.py (functions: create_extraction_task, etc.)
2. ✅ Configure crew in main.py (lines 247-252)
3. ✅ Set up sequential workflow (Process.sequential)
4. ✅ Agent handoffs working
5. ⚠️ Integration tests - NOT YET IMPLEMENTED

#### Phase 6: Main Application ✅ (COMPLETE)
1. ✅ Implement main.py entry point (run_invoice_analysis function)
2. ✅ Add CLI interface (argparse - lines 275-312)
3. ⚠️ Create sample input files - Placeholder READMEs only
4. ⚠️ Write end-to-end tests - NOT YET IMPLEMENTED
5. ✅ Create usage examples (in gen_z_agent/README.md)

#### Phase 7: Communication & Reports ✅ (COMPLETE - Template Stage)
1. ⚠️ Implement email sending - Template-based, not fully functional
2. ⚠️ Implement Slack notifications - Template-based, not fully functional
3. ✅ Create report templates (in config.py)
4. ⚠️ Test output formats - Agents output instructions, not actual files yet

#### Phase 8: Documentation ✅ (PARTIALLY COMPLETE)
1. ✅ Write comprehensive docs/ - Foundry integration docs added
2. ⚠️ Create tutorial examples - Basic examples in README
3. ⚠️ Add error handling improvements - Basic error handling present
4. ⚠️ Performance optimization - Not yet addressed
5. ⚠️ Security review - Not yet performed

### 🚧 INCOMPLETE PHASES

#### Phase 2: Data Models ⚠️ (NOT STARTED)
1. ❌ Define Pydantic models for electoral data
2. ❌ Create validation schemas
3. ❌ Write model unit tests
4. ❌ Document data structures

**Current Approach**: Agents work with unstructured data (dicts/JSON)
**Priority**: Low - System is functional without strict schemas

#### Phase 3: Tools & Utilities ⚠️ (NOT STARTED)
1. ❌ Implement PDF parser (pdf_parser.py) - Using FileReadTool instead
2. ❌ Implement HTML parser (html_parser.py)
3. ❌ Create Korean text OCR tool (korean_ocr.py)
4. ❌ Build data enrichment utilities
5. ❌ Write tool unit tests

**Current Approach**: Using CrewAI's built-in FileReadTool
**Priority**: Medium - Custom tools would improve PDF/HTML parsing accuracy

### 🎯 NEXT PRIORITIES (Recommended Order)

#### Priority 1: Testing Infrastructure (HIGH)
**Rationale**: Production system needs test coverage
1. Create tests/ directory structure
2. Add pytest.ini configuration
3. Write unit tests for agents
4. Write integration tests for workflow
5. Add fixtures with sample election data
6. Set up pytest and coverage in requirements-dev.txt

**Estimated Effort**: 2-3 days
**Impact**: High - Ensures reliability and prevents regressions

#### Priority 2: Refactor to Modular Architecture (MEDIUM)
**Rationale**: Improve maintainability as system grows
1. Create gen_z_agent/agents/ directory
2. Extract each agent to separate file
3. Create gen_z_agent/tasks/ directory
4. Extract task definitions
5. Create gen_z_agent/crew.py for crew configuration
6. Update imports in main.py

**Estimated Effort**: 1-2 days
**Impact**: Medium - Better organization, easier to maintain
**Blocker**: None, can be done anytime
**Trigger**: When main.py exceeds 500 lines or team grows

#### Priority 3: Custom PDF/HTML Tools (MEDIUM)
**Rationale**: Improve Korean election document parsing
1. Implement pdf_parser.py with pdfplumber
2. Implement html_parser.py with BeautifulSoup
3. Add Korean OCR support (tesseract-ocr)
4. Create data enrichment utilities
5. Write tool unit tests

**Estimated Effort**: 3-4 days
**Impact**: High - Better extraction accuracy for Korean text
**Blocker**: Need sample Korean election PDFs for testing

#### Priority 4: CI/CD Pipeline (LOW-MEDIUM)
**Rationale**: Automate testing and deployment
1. Create .github/workflows/ directory
2. Add pytest workflow
3. Add code quality checks (black, flake8)
4. Add coverage reporting
5. Add automated releases

**Estimated Effort**: 1 day
**Impact**: Medium - Improves development workflow
**Blocker**: Priority 1 (testing) should be completed first

#### Priority 5: Enhanced Reporting (LOW)
**Rationale**: Produce actual Excel/PDF files, not just instructions
1. Implement actual Excel generation with openpyxl
2. Implement PDF generation with reportlab
3. Add chart generation with matplotlib
4. Implement email sending with smtplib
5. Implement Slack posting with webhooks

**Estimated Effort**: 2-3 days
**Impact**: High - Completes the full automation workflow
**Current State**: Agents output Markdown instructions for reports

#### Priority 6: Palantir Foundry Integration Implementation (LOW)
**Rationale**: Documentation exists, implementation could follow
1. Set up Azure Event Hubs connection
2. Implement Bronze layer ingestion
3. Create Silver layer transformations
4. Build Gold layer aggregations
5. Deploy to Foundry instance

**Estimated Effort**: 1-2 weeks
**Impact**: Low for Gen Z Agent, High for healthcare analytics
**Blocker**: Requires Azure and Foundry infrastructure access
**Note**: This is a separate use case from Korean election analysis

### 📊 Current System Maturity

| Component | Status | Completeness | Notes |
|-----------|--------|--------------|-------|
| Agent Implementation | ✅ Complete | 100% | All 5 agents functional |
| Configuration | ✅ Complete | 100% | Comprehensive config.py |
| CLI Interface | ✅ Complete | 100% | Argparse with all options |
| Task Definitions | ✅ Complete | 100% | All task functions implemented |
| CrewAI Integration | ✅ Complete | 100% | Sequential workflow working |
| Korean Election Support | ✅ Complete | 100% | 5 candidates, vote types configured |
| Testing | ❌ Not Started | 0% | No tests exist |
| Modular Architecture | ❌ Not Started | 0% | Monolithic main.py |
| Custom Tools | ❌ Not Started | 0% | Using built-in tools only |
| Report Generation | ⚠️ Partial | 30% | Templates exist, no actual file output |
| Email/Slack | ⚠️ Partial | 20% | Templates exist, not functional |
| Documentation | ✅ Good | 80% | README, CLAUDE.md, Foundry docs |

**Overall Maturity**: 🟢 **Production-Ready for Manual Execution** (60% complete)
- Core functionality works end-to-end
- Suitable for demo and manual analysis workflows
- Needs testing and automation for production deployment

## 🤝 AI Assistant Guidelines

### When Working on This Project

#### Understanding Context
1. **Korean Language**: This project deals with Korean electoral data. Familiarize yourself with Korean election terminology (개표상황표 = election count sheet, 득표수 = vote count, 후보자 = candidate, etc.)
2. **Domain Knowledge**: Understand basic electoral data concepts (turnout, margins, voting districts)
3. **Multi-Agent Systems**: Understand how CrewAI orchestrates multiple agents

#### Making Changes
1. **Read First**: Always read existing files before making changes
2. **Follow Structure**: Adhere to the intended project structure
3. **Test Coverage**: Write tests for new functionality
4. **Documentation**: Update docstrings and CLAUDE.md when adding features
5. **Incremental**: Make small, focused commits with clear messages

#### Code Generation
1. **Type Hints**: Always include type hints in function signatures
2. **Error Handling**: Wrap risky operations in try-except blocks
3. **Validation**: Validate inputs, especially file paths and API responses
4. **Logging**: Add appropriate logging statements
5. **Comments**: Explain complex logic, especially Korean-specific parsing

#### Agent Development
When creating or modifying agents:
1. **Role Definition**: Clearly define the agent's role and goal
2. **Tools**: Identify what tools the agent needs
3. **Backstory**: Write a compelling backstory for better performance
4. **Task Output**: Define expected output format (use Pydantic models)
5. **Dependencies**: Document what data the agent expects from previous agents

Example agent definition:
```python
from crewai import Agent
from gen_z_agent.tools.pdf_parser import PDFParserTool
from gen_z_agent.utils.config import Config

extractor_agent = Agent(
    role="Invoice Data Extractor",
    goal="Extract structured data from Korean election count sheets",
    backstory="""You are an expert in reading and parsing Korean electoral
    documents. You understand the structure of 개표상황표 (election count sheets)
    and can accurately extract candidate names, vote counts, and regional data
    even from complex PDF layouts.""",
    tools=[PDFParserTool()],
    verbose=True,
    allow_delegation=False,
    llm=claude_llm
)
```

#### Testing Strategy
1. **Unit Tests**: Test individual functions with mock data
2. **Integration Tests**: Test agent interactions with real-like data
3. **Fixtures**: Create realistic test fixtures (sample PDFs, HTML)
4. **Edge Cases**: Test error conditions, malformed input, missing data
5. **Korean Text**: Test with actual Korean text, not just ASCII

#### Git Workflow
1. **Branch**: Work on feature branch `claude/claude-md-mi7hteawd991ao45-01SgHLwx2Yq6MtMrDbeik7ME`
2. **Commit Messages**: Use conventional commits (feat:, fix:, docs:, test:, refactor:)
3. **Small Commits**: Commit logical units of work
4. **Push Regularly**: Push to remote after completing features

Example commit messages:
```
feat: implement PDF parser for Korean election data
fix: handle missing candidate names in validation
docs: add agent architecture documentation
test: add integration tests for extractor agent
refactor: simplify error handling in validator
```

#### Common Pitfalls to Avoid
1. ❌ Don't hardcode API keys (use environment variables)
2. ❌ Don't use print() for logging (use logging module)
3. ❌ Don't ignore exceptions silently (log and handle properly)
4. ❌ Don't create files without checking if they exist (user confirmation)
5. ❌ Don't assume file encoding (detect or default to UTF-8)
6. ❌ Don't skip input validation (validate early, fail fast)
7. ❌ Don't duplicate code (create utility functions)
8. ❌ Don't mix Korean and English randomly (be consistent within a file)

#### Korean Text Handling
- **Encoding**: Always use UTF-8
- **Libraries**: Use appropriate libraries (konlpy for NLP, not needed for simple parsing)
- **Regex**: Be careful with Korean character ranges (가-힣)
- **Normalization**: Be aware of Korean Unicode normalization issues (NFC vs NFD)

Example:
```python
import re

def is_korean_name(text: str) -> bool:
    """Check if text appears to be a Korean name."""
    # Korean names are typically 2-4 characters, all Hangul
    return bool(re.match(r'^[가-힣]{2,4}$', text))

def clean_korean_text(text: str) -> str:
    """Clean and normalize Korean text."""
    import unicodedata
    # Normalize to NFC (most common form for Korean)
    text = unicodedata.normalize('NFC', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()
```

#### Performance Considerations
1. **Batch Processing**: Process multiple documents in parallel when possible
2. **Caching**: Cache parsed data to avoid re-parsing
3. **Async Operations**: Use async for I/O operations (API calls, file reads)
4. **Memory**: Be mindful of memory with large PDFs (stream processing)
5. **Rate Limiting**: Respect API rate limits (Claude, Serper, etc.)

#### Security Considerations
1. **Input Validation**: Sanitize all user inputs and file paths
2. **API Keys**: Never log or expose API keys
3. **File Access**: Validate file paths to prevent directory traversal
4. **Dependencies**: Keep dependencies updated for security patches
5. **Data Privacy**: Be careful with sensitive electoral data

## 📚 Key Resources

### External Documentation
- [CrewAI Documentation](https://docs.crewai.com/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Korean Election Commission](https://www.nec.go.kr/) - For understanding data formats

### Internal References
- README.md - User-facing project overview
- env.example - Required environment variables
- docs/architecture.md - (TODO) Detailed architecture documentation
- docs/agent_design.md - (TODO) Individual agent design docs

## 🔄 Updating This Document

This CLAUDE.md file should be updated when:
1. **Project structure changes** - New directories, major refactoring
2. **New agents or tools added** - Document their purpose and usage
3. **Workflow changes** - Updated agent interaction patterns
4. **New conventions adopted** - Code style, testing practices
5. **Dependencies change** - Major version updates, new libraries
6. **Implementation priorities shift** - Adjust phase plans

### Update Checklist
When updating CLAUDE.md:
- [ ] Update table of contents if sections change
- [ ] Update file structure diagrams
- [ ] Update code examples to match current implementation
- [ ] Update dependency versions
- [ ] Add examples for new patterns or conventions
- [ ] Update "Current Status" section
- [ ] Document any breaking changes
- [ ] Add date of last update below

**Last Updated**: 2025-11-20
**Last Updated By**: Claude (Anthropic AI Assistant)
**Version**: 2.0.0 - Comprehensive update reflecting actual implementation state, added Foundry integration docs, updated priorities

---

## 📝 Quick Reference

### Essential Commands
```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
cp env.example .env
# Edit .env with your API keys

# Run
python gen_z_agent/main.py --input <pdf_file>

# Test
pytest                          # Run all tests
pytest -v                       # Verbose output
pytest --cov=gen_z_agent       # With coverage
pytest tests/test_agents/       # Specific directory

# Code Quality
black gen_z_agent/             # Format code
flake8 gen_z_agent/            # Lint
isort gen_z_agent/             # Sort imports
mypy gen_z_agent/              # Type check
```

### Common Tasks for AI Assistants

**Task: "Implement the PDF Parser Tool"**
1. Read existing tools in `gen_z_agent/tools/`
2. Create `gen_z_agent/tools/pdf_parser.py`
3. Implement using pdfplumber with Korean text support
4. Add error handling and logging
5. Write tests in `tests/test_tools/test_pdf_parser.py`
6. Update this CLAUDE.md if needed

**Task: "Add a new agent"**
1. Review the 5 agent architecture above
2. Create agent file in `gen_z_agent/agents/`
3. Define role, goal, backstory, tools
4. Implement agent logic
5. Add to crew configuration in `crew.py`
6. Write integration tests
7. Document in CLAUDE.md

**Task: "Fix a bug"**
1. Read the relevant code files
2. Understand the error (check logs)
3. Write a failing test that reproduces the bug
4. Fix the bug
5. Ensure test passes
6. Add logging if needed
7. Commit with "fix:" prefix

**Task: "Add Korean text support"**
1. Check encoding (UTF-8)
2. Test with actual Korean text
3. Use proper regex for Hangul (가-힣)
4. Normalize Unicode (NFC)
5. Add test with Korean fixtures
6. Document any Korean-specific logic

---

## 🎓 Learning Resources for AI Assistants

If you need to understand more about the technologies used:

1. **CrewAI**: Multi-agent orchestration framework
   - Sequential vs Parallel task execution
   - Agent delegation and communication
   - Tool creation and usage

2. **Anthropic Claude**: Large language model API
   - Prompt engineering for agents
   - Token management
   - Streaming vs non-streaming

3. **Korean Text Processing**:
   - Hangul Unicode ranges
   - Korean name patterns
   - Common electoral terminology

4. **Electoral Data Analysis**:
   - Vote counting verification
   - Statistical anomaly detection
   - Turnout analysis
   - Regional comparisons

---

**End of CLAUDE.md**

For questions or clarifications, please ask the human developer or refer to the README.md for user-facing documentation.
