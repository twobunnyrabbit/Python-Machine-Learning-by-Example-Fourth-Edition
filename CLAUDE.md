# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python machine learning educational repository containing code examples from "Python Machine Learning by Example, Fourth Edition" (Packt Publishing). The repository has been modified to use `uv` for modern Python package management and includes both the original book examples and additional educational materials.

## Development Environment Setup

### Package Management
This project uses `uv` for Python package management:
- Install dependencies: `uv sync`
- Add new packages: `uv add <package_name>`
- For PyTorch specifically: `uv pip install torch torchvision` (due to installation requirements)
- Python version: >=3.10.17 (specified in pyproject.toml)

### Key Dependencies
- Core ML libraries: scikit-learn, numpy, pandas, matplotlib, seaborn
- Advanced libraries: scipy, plotly, plotnine
- Data sources: kaggle, kagglehub, ucimlrepo
- Additional: akde, openpyxl, pyhere

## Repository Structure

The repository is organized by book chapters (ch2/ through ch15/), each containing:
- **Jupyter notebooks** (.ipynb): Interactive examples with explanations
- **Python scripts** (.py): Converted versions of notebooks for direct execution
- **Quarto documents** (.qmd): Additional documentation and analysis
- **Data files**: Chapter-specific datasets in data/ directory

### Chapter Organization
- ch2/: Naive Bayes (movie recommendation, spam detection, heart disease)
- ch3/: Decision Trees (ad click-through prediction)
- ch4-15/: Various ML algorithms and applications

## Working with Code

### Preferred Development Environment
- **Primary format**: Quarto documents (.qmd) for faster development with Posit's Positron IDE
- **Legacy formats**: Jupyter notebooks (.ipynb) and Python scripts (.py) from original repository
- Repository includes all three formats for flexibility

### Document Conversion
Convert between formats as needed:
```bash
# Convert notebook to Quarto
quarto convert <notebook>.ipynb
# Convert notebook to Python script
jupyter nbconvert --to python <notebook>.ipynb --TemplateExporter.exclude_input_prompt=True
```

### Code Style
- All Python files include proper headers with chapter information and author credits
- Functions include comprehensive docstrings with parameter and return type descriptions
- Code follows educational patterns with clear variable names and step-by-step implementations

### Data Access
- MovieLens dataset (ml-1m/) in data/ch02/ for recommendation examples
- CSV files for classification examples in respective chapter directories
- External datasets accessed via kaggle/ucimlrepo APIs

## ML Implementation Patterns

### From-Scratch Implementations
Many algorithms are implemented from scratch for educational purposes:
- Custom functions for calculating priors, likelihoods, and posteriors
- Manual implementation of metrics (Gini impurity, entropy)
- Step-by-step probability calculations

### Scikit-learn Integration
Each chapter typically includes both custom implementations and scikit-learn equivalents for comparison:
- Custom implementation followed by sklearn version
- Performance and accuracy comparisons
- Validation of custom algorithms against established libraries

## Development Workflow

1. **Prefer Quarto documents (.qmd)** for new work - faster development with Positron IDE
2. Use existing chapter structure - avoid creating new top-level directories  
3. Follow the educational pattern: theory → custom implementation → sklearn comparison
4. Convert between .qmd/.ipynb/.py formats as needed for compatibility
5. Include proper attribution and chapter references in all new code