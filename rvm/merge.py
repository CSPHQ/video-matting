import cv2
import moviepy.editor as mpe


def combine_audio(vidname, audname, outname):
    cap = cv2.VideoCapture(vidname)
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    my_clip = mpe.VideoFileClip(vidname)
    audio_background = mpe.AudioFileClip(audname)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(outname, fps=fps)


if __name__ == '__main__':
    import fire
    fire.Fire(combine_audio)
