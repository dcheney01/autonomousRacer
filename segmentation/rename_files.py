#rename files to have 00000 instead of 0 based on number in file
import os

path = './fourth_lap/rgb/'
images = os.listdir(path)
images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
for i, file in enumerate(images):
    os.rename(path + file, path + f'{i:05d}.jpg')
    print(f'{i:05d}.jpg')



