#!/bin/bash

set -e

docker run --rm -it -v "$(pwd):/src/" --entrypoint /bin/bash cdrx/pyinstaller-windows -c "python -m pip install --upgrade pip && python -m pip install --upgrade pyinstaller && pip install -r requirements.txt && pyinstaller --onefile --clean -y --dist ./dist/windows --workpath /tmp *.spec"

sudo chown -R qhduan dist/
cp video_matting/rvm_mobilenetv3_fp32.onnx dist/windows/
(cd dist/windows/ && zip -r ../video_matting_cli.zip ./*)
