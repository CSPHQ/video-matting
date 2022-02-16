
import os
import cv2
import numpy as np
from video_matting.create_model import create_model_for_provider
from video_matting.rvm import compute_without_border


CURRENT_DIR = os.path.realpath(os.path.dirname(__file__))


def camera(
    model_path='rvm_mobilenetv3_fp32.onnx',
    downsample=0.5,
    green_color=[0, 255, 0],
    num_threads=None,
    camera=0,
    width=320,
    height=180,
):
    """
    Run a camera window to show video matting, useful when you use OBS to live stream.
    Args:
        model_path: path to the model
        downsample: downsample factor, default is 0.5, slower is faster and lower quality
        green_color: color of the background, default green: 0, 255, 0
        num_threads: number of threads to use for inference, default is None, will use 4 threads
        camera: camera index, default is 0
        width: width of the image, default is 320
        height: height of the image, default is 180
    """

    if os.path.exists(model_path):
        pass
    elif os.path.exists(os.path.join(CURRENT_DIR, model_path)):
        model_path = os.path.join(CURRENT_DIR, model_path)
    assert os.path.exists(model_path), 'Model not found'

    vid = cv2.VideoCapture(camera)

    sess = create_model_for_provider(model_path, num_threads=num_threads)
    green = np.array(green_color).reshape([1, 1, 3]) / 255.
    rec = [np.zeros([1, 1, 1, 1], dtype=np.float32)] * 4  # Must match dtype of the model.
    downsample_ratio = np.array([downsample], dtype=np.float32)  # dtype always FP32

    while(True):
        _, frame = vid.read()
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        src = np.expand_dims(np.transpose(frame, (2, 0, 1)), 0).astype(np.float32) / 255.0
        batch_inputs = src
        if 'fp16' in model_path:
            batch_inputs = batch_inputs.astype('float16')
            rec = [x.astype('float16') for x in rec]
        elif 'fp32' in model_path:
            batch_inputs = batch_inputs.astype('float32')
            rec = [x.astype('float32') for x in rec]

        fgr, pha, *rec = sess.run([], {
            'src': batch_inputs,
            'r1i': rec[0][-1:], 'r2i': rec[1][-1:], 'r3i': rec[2][-1:], 'r4i': rec[3][-1:],
            'downsample_ratio': downsample_ratio
        })

        fgr = np.transpose(fgr, [0, 2, 3, 1])
        pha = np.transpose(pha, [0, 2, 3, 1])
        output_img = np.array(compute_without_border(fgr, pha, green)).astype('uint8')
        oi = output_img[0]

        oi = cv2.cvtColor(oi, cv2.COLOR_RGB2BGR)
        cv2.imshow('camera', oi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


def cmd():
    from fire import Fire
    Fire(camera)


if __name__ == '__main__':
    cmd()
