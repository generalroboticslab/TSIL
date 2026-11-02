# Validation

Tested on Linux with Python 3.8 and PyTorch 2.4.1. MiniGrid supports CPU and
GPU; MetaWorld requires an NVIDIA GPU and Isaac Gym.

After installation, run these checks from the repository root:

```bash
python -m pip check
python tests/verify_package.py
python tests/verify_entrypoints.py
python tests/verify_plotting.py
python tests/verify_report_aggregation.py
```

To check the notebook implementation:

```bash
python -m pip install -r notebooks/requirements.txt
python tests/verify_notebook.py
```

Use `--smoke` with a training command for a short run on your machine.
