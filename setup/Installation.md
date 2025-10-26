### Create and activate virtual environment
```bash
python3.10 -m venv simcalkv
source simcalkv/bin/activate
```

### Use a mirror site if necessary
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

### Installation（transformers==4.46.3）
```bash
# basic package
pip install torch numpy matplotlib seaborn pandas transformers datasets evaluate -i https://pypi.tuna.tsinghua.edu.cn/simple

# pillow
pip install --upgrade pillow -i https://pypi.tuna.tsinghua.edu.cn/simple

# accelerate
pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple

# absl
pip install absl-py -i https://pypi.tuna.tsinghua.edu.cn/simple

# nltk
pip install nltk -i https://pypi.tuna.tsinghua.edu.cn/simple

# rouge-score
pip install rouge-score -i https://pypi.tuna.tsinghua.edu.cn/simple
```

