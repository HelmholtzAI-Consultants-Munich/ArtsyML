from setuptools import setup
from pathlib import Path

setup(
      name = 'artsyml',
      install_requires=[
          l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
      ]
)
