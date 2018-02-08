import os


ckptdir = 'dir/'
path = os.path.join(os.getcwd(), ckptdir)

assert len(os.listdir(path))>0

