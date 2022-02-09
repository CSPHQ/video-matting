
import os
import math
import tempfile
import fire
from tqdm import tqdm
import numpy as np
import cv2

from video_matting.create_model import create_model_for_provider
from video_matting.merge import combine_audio
from video_matting.animegan import AnimeGAN

CURRENT_DIR = os.path.realpath(os.path.dirname(__file__))

try:
    from jax import jit
    USE_JIT = True
    print('use jit')
except:  # noqa
    USE_JIT = False
    print('jit not work')


def get_video(input_file):
    cap = cv2.VideoCapture(input_file)
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                src = np.expand_dims(np.transpose(frame, (2, 0, 1)), 0).astype(np.float32) / 255.0
                yield src
        else:
            break
    cap.release()


def compute_border(fgr, pha, border):
    output_img_border = fgr * pha + (1 - pha) * border
    output_img_border = np.clip(output_img_border, 0.0, 1.0)
    output_img_border = (output_img_border * 255.0).astype('uint8')
    ipha = ((pha > 0.2) * 255).astype('uint8')
    return output_img_border, ipha


def compute_border_2(output_img_border, img_dilation_filter, green):
    return (output_img_border * img_dilation_filter) + (1 - img_dilation_filter) * green * 255


def compute_without_border(fgr, pha, green):
    output_img = fgr * pha + (1 - pha) * green
    output_img = np.clip(output_img, 0.0, 1.0)
    output_img = (output_img * 255.0)
    return output_img


if USE_JIT:
    compute_border = jit(compute_border)
    compute_border_2 = jit(compute_border_2)
    compute_without_border = jit(compute_without_border)


def write_frame(fgr, pha, border, green, use_border):
    fgr = np.transpose(fgr, [0, 2, 3, 1])
    pha = np.transpose(pha, [0, 2, 3, 1])
    if use_border:
        output_img_border, ipha = compute_border(fgr, pha, border)
        output_img_border = np.array(output_img_border)
        ipha = np.array(ipha)

        dilation = cv2.dilate(np.array(ipha[0]), np.ones((5, 5)), iterations=1)
        img_dilation = np.expand_dims(np.expand_dims(dilation, 0), -1)
        img_dilation_filter = img_dilation.astype('float32') / 255.0

        output_img_border = np.array(
            compute_border_2(
                output_img_border, img_dilation_filter, green
            )
        ).astype('uint8')

        output_img = output_img_border
    else:
        output_img = np.array(compute_without_border(fgr, pha, green)).astype('uint8')
    oi = cv2.cvtColor(output_img[0], cv2.COLOR_RGB2BGR)
    return oi


def generate_result(input_file, all_frames, sess, model_path, downsample):
    pbar = tqdm(
        get_video(input_file=input_file),
        total=math.ceil(all_frames)
    )
    rec = [ np.zeros([1, 1, 1, 1], dtype=np.float32) ] * 4  # Must match dtype of the model.
    downsample_ratio = np.array([downsample], dtype=np.float32)  # dtype always FP32
    for src in pbar:
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
        # import pdb; pdb.set_trace()
        yield fgr, pha


def convert(
    input_file,
    output_file,
    model_path='rvm_mobilenetv3_fp32.onnx',
    downsample=0.5,
    green_color=[0, 255, 0],
    use_border=False,
    border_color=[255, 255, 255],
    num_threads=None,
    animegan=None
):

    if os.path.exists(model_path):
        pass
    elif os.path.exists(os.path.join(CURRENT_DIR, model_path)):
        model_path = os.path.join(CURRENT_DIR, model_path)

    assert os.path.exists(input_file), 'Input file not found'
    assert os.path.exists(model_path), 'Model not found'

    sess = create_model_for_provider(model_path, num_threads=num_threads)

    green = np.array(green_color).reshape([1, 1, 3]) / 255.
    border = np.array(border_color).reshape([1, 1, 3]) / 255.

    cap = cv2.VideoCapture(input_file)
    all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    f = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    if animegan is not None:
        agan = AnimeGAN(animegan)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f.name, fourcc, fps, (width, height))
    for fgr, pha in generate_result(input_file, all_frames, sess, model_path, downsample):
        if animegan is not None:
            fgr = agan(fgr)
        out.write(write_frame(fgr, pha, border, green, use_border))
    out.release()

    print(f'start combine converted video and audio from original video into {output_file}')
    combine_audio(f.name, input_file, output_file)
    if os.path.exists(f.name):
        try:
            os.remove(f.name)
        except:  # noqa
            pass
    print('DONE')


if __name__ == '__main__':
    fire.Fire(convert)
