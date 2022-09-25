#!/usr/bin/env python3

import os
from PIL import Image


def main():
    #dirname = "/home/kiru/Projects/histre/backend/txt2img_output/"
    dirname = "/db/output/txt2img/static"
    for f in os.listdir(dirname):
        if f.endswith(".png"):
            filename = os.path.join(dirname, f)
            print(f"Processing {filename}")
            filename_jpg_template = filename.split(".")[0] + "-{}.jpg"
            im = Image.open(filename).convert("RGB")
            im.save(filename_jpg_template.format(512))
            im.resize((256,256)).save(filename_jpg_template.format(256))


if __name__ == "__main__":
    main()
