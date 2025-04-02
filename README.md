# Staged-Approach-PRJ

This folder contains all the code necessary to run the **Staged Approach** experiment as part of my 6CCS3PRJ Final Year Project.

---

## Setup Instructions

### 1. Python Version

Ensure you have **Python 3.8 or newer** installed.

### 2. Create a Virtual Environment (Recommended)

```bash
# Create the environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
## Running the Experiment
Make sure you are inside the repository root (`cd Staged-Approach-PRJ` or the renamed version):

```bash
cd "Staged-Approach-PRJ"
```
or 
```bash
cd "Staged Approach"
```

Run the benchmark script to execute all staged experiments at once:

```bash
python scripts/benchmark.py
```

or (**more likely to succeed**)

```bash
python -m scripts.benchmark
```

## Experiment Outputs

Trained models are saved as `.pth` files in the project root:

- `single_shot_transformer.pth`
- `staged_transformer.pth`
- `staged2_transformer.pth`
- `staged3_transformer.pth`
- `staged4_transformer.pth`
- `staged5_transformer.pth`

Perplexity scores for each model will be printed to the terminal during evaluation.

## Experiment Outputs

To change the number of layers and attention heads per layer, edit the following file:

```bash
./models/child_transformer.py
```

Modify the values of `n_layer` and `n_head` to your desired configuration.

## ⚠️ Disclaimer

**Note:** Results obtained from running this code may differ from the results reported in the final project report due to hardware differences and the inherent nature of some non deterministic GPU operations.

## Complete Dependency List

Expand below to view all required packages with exact versions:

<details>
  <summary>Full list of required Python packages</summary>

  ```text
  accelerate==1.2.1
  aiohappyeyeballs==2.4.4
  aiohttp==3.11.11
  aiosignal==1.3.2
  attrs==24.3.0
  certifi==2024.12.14
  charset-normalizer==3.4.1
  datasets==3.2.0
  dill==0.3.8
  evaluate==0.4.3
  filelock==3.16.1
  frozenlist==1.5.0
  fsspec==2024.9.0
  huggingface-hub==0.27.1
  idna==3.10
  inquirerpy==0.3.4
  Jinja2==3.1.5
  joblib==1.4.2
  MarkupSafe==3.0.2
  mpmath==1.3.0
  multidict==6.1.0
  multiprocess==0.70.16
  networkx==3.4.2
  numpy==2.2.1
  packaging==24.2
  pandas==2.2.3
  pfzy==0.3.4
  pip==24.0
  prompt_toolkit==3.0.50
  propcache==0.2.1
  psutil==6.1.1
  pyarrow==18.1.0
  python-dateutil==2.9.0.post0
  pytz==2024.2
  PyYAML==6.0.2
  regex==2024.11.6
  requests==2.32.3
  safetensors==0.5.1
  scikit-learn==1.6.1
  scipy==1.15.2
  setuptools==65.5.0
  six==1.17.0
  sympy==1.13.1
  threadpoolctl==3.5.0
  tokenizers==0.21.0
  torch==2.5.1
  tqdm==4.67.1
  transformers==4.47.1
  typing_extensions==4.12.2
  tzdata==2024.2
  urllib3==2.3.0
  wcwidth==0.2.13
  xxhash==3.5.0
  yarl==1.18.3
```
</details>
