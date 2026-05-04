import conf.global_vars as glob_vars


class Settings:
    def __init__(self, glob_vars):

        for attr in dir(glob_vars):
            if attr.isupper():
                setattr(self, attr, getattr(glob_vars, attr))


settings = Settings(glob_vars)