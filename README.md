# Video matting

model from [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting/)

install

```
pip install video-matting
```

usage

```
video-matting input.mp4 output.mp4
```


## For mac m1

python==3.9

```
pip install video-matting --no-deps
pip install https://test-files.pythonhosted.org/packages/79/20/47672664090294646b621dacf8d4e5d43e085d0b8c2797677b08c3126534/onnxruntime-1.9.0.dev174552-cp39-cp39-macosx_10_14_universal2.whl
pip install https://storage.googleapis.com/jax-releases/mac/jaxlib-0.1.74-cp39-none-macosx_11_0_arm64.whl
pip install moviepy tqdm jax fire numpy opencv-python
```

## generate win exe

from https://github.com/cdrx/docker-pyinstaller

```
docker run -v "$(pwd):/src/" -it --entrypoint /bin/bash cdrx/pyinstaller-linux -c "pyinstaller --onefile video_matting/video_matting_cli.py"
sudo chown -R qhduan *.spec
```

```
docker run --rm -it -v "$(pwd):/src/" --entrypoint /bin/bash cdrx/pyinstaller-windows -c "python -m pip install --upgrade pip && pip install -U pyinstaller && pip install -r requirements.txt && pyinstaller --onefile --clean -y --dist ./dist/windows --workpath /tmp *.spec"

sudo chown -R qhduan dist/ && cp video_matting/rvm_mobilenetv3_fp32.onnx dist/windows/ && (cd dist/windows/ && zip -r ../video_matting_cli.zip ./*)
```
