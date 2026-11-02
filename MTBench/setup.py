"""Install the MTBench environment subset used by TSIL."""

from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "gym==0.23.1",
    "hydra-core>=1.1",
    "jinja2",
    "matplotlib",
    "numpy",
    "omegaconf",
    "opencv-python",
    "pyfqmr",
    "pyvirtualdisplay",
    "scipy",
    "skrl",
    "termcolor",
    "torch",
    "torchvision",
    "tqdm",
    "trimesh",
    "wandb",
]


setup(
    name="temporalsil-mtbench-envs",
    version="0.1.0",
    description="MTBench Isaac Gym environment subset used by TSIL.",
    keywords=["robotics", "reinforcement-learning", "isaacgym", "temporalsil"],
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    zip_safe=False,
)
