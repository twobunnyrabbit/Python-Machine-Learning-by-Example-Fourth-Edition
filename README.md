# Python-Machine-Learning-by-Example-Fourth-Edition
Python Machine Learning by Example, Fourth Edition

I forked this from the original repo on 2025-04-15 and added the following:
- converted it to use `uv` for package installation
- most dependencies are installed via `uv add <package_name> ...`
- `pytorch` was not able to be installed this way, instead:
    `uv pip install torch torchvision`

## Dataset Configuration

Datasets are stored externally and accessed via a global configuration system using environment variables.

### Setup Instructions

1. **Copy datasets** to your local machine (from OneDrive or other source)
2. **Create a `.env` file** in the repository root with your dataset paths:
   ```env
   # Dataset paths configuration
   DATA_ROOT=/path/to/your/datasets/Python-Machine-Learning-by-Example-Fourth-Edition/data
   MOVIELENS_PATH=${DATA_ROOT}/ch02/ml-1m
   CLICK_RATE_PATH=${DATA_ROOT}/ch03
   ```
3. **Install dependencies** including python-dotenv:
   ```bash
   uv sync
   ```

### Usage in Code

All Python files can access datasets using the global config:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import get_movielens_file, get_data_path

# Use configured dataset paths
data_path = get_movielens_file('ratings.dat')
df = pd.read_csv(data_path, ...)
```

**Note:** The `.env` file is not committed to the repository for security and portability reasons.
