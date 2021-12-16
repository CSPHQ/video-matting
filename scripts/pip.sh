#!/bin/bash
# python -m pip install -U twine keyring keyrings.alt

set -e
rm -rf ./dist ./build ./*.egg-info
python3 setup.py bdist_wheel
twine upload dist/*.whl

