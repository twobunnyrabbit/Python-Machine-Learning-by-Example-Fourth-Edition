# Python-Machine-Learning-by-Example-Fourth-Edition
Python Machine Learning by Example, Fourth Edition

I forked this from the original repo on 2025-04-15 and added the following:
- converted it to use `uv` for package installation
- most dependencies are installed via `uv add <package_name> ...`
- `pytorch` was not able to be installed this way, instead:
    `uv pip install torch torchvision`

