import os
import glob
import shutil

def extract_id(string):
    splitter = string.split('/')
    splitter = splitter[-1].split('_')
    splitter = splitter[0]

    return splitter
files = glob.glob('/home/def/Music/gallery/*.jpg')

for file in files:
    id_photo = extract_id(file)
    if os.path.exists('/home/def/Music/gallery_for_dataset/' + id_photo):
        shutil.copy(file, '/home/def/Music/gallery_for_dataset/' + id_photo)
    else:
        os.mkdir('/home/def/Music/gallery_for_dataset/' + id_photo)
        shutil.copy(file, '/home/def/Music/gallery_for_dataset/' + id_photo)


