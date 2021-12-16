import fire
from video_matting.rvm import convert


def cmd():
    fire.Fire(convert)


if __name__ == '__main__':
    cmd()
