
import numpy as np
from video_matting.create_model import create_model_for_provider


class AnimeGAN:
    def __init__(self, model_path='face_paint_512_v2.onnx'):
        self.sess = create_model_for_provider(model_path)
    
    def __call__(self, img):
        # img = Image.open('lbb.webp')
        img_data = np.array(img)
        if len(img_data.shape) != 4:
            if img_data.shape[0] != 3:
                img_data = np.transpose(img_data, (2, 0, 1))
            img_data = np.expand_dims(img_data, 0)
        # img_data = img_data.astype('float32') / 255.0
        # img_data = img_data[:, :3, :, :]
        outs = self.sess.run(None, {
            self.sess.get_inputs()[0].name: img_data,
        })
        img_out = outs[0]
        img_out = np.clip(img_out, 0.0, 1.0)  #  * 255.0
        # img_out = np.transpose(np.squeeze(outs[0]), (1, 2, 0))
        # img_output = Image.fromarray(img_out.astype('uint8'))
        return img_out


if __name__ == '__main__':
    a = AnimeGAN()
