#rename files to have 00000 instead of 0 based on number in file

import os

path = './tmp/labels/lap3/'
images = os.listdir(path)
images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
for i, file in enumerate(images):
    os.rename(path + file, path + "lap3_" + file)


# path = './tmp/images/lap3/'
# images = os.listdir(path)
# images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
# for i, file in enumerate(images):
#     os.rename(path + file, path + "lap3_" + f'{i:05d}.jpg')
