import cv2
# due to
# https://stackoverflow.com/questions/44615249/attributeerror-module-object-has-no-attribute-audio-fadein
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip


def combine_audio(vidname, audname, outname):
    cap = cv2.VideoCapture(vidname)
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    my_clip = VideoFileClip(vidname)
    audio_background = AudioFileClip(audname)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(outname, fps=fps, codec="libx264", audio_codec='aac')


if __name__ == '__main__':
    import fire
    fire.Fire(combine_audio)
