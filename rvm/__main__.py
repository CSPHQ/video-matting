import fire
from .rvm import convert


def cmd():
    fire.Fire(convert)

    
if __name__ == '__main__':
    cmd()
