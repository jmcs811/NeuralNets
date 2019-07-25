# used to move and unzip the malware files from 
# theZoo directory.

import os
import shutil
import sys
from zipfile import ZipFile

# command line args
# first is where malware images are
# second is where you want to unzipped files to go
src = sys.argv[1]
dst = sys.argv[2]

for root, subdirs, files in os.walk(src):
    for filename in files:
        file_path = os.path.join(root, filename)
        if (filename.endswith("zip")):
            dst_path = os.path.join(dst, filename)
            try:
                with ZipFile(file_path) as zf:
                    zf.extractall(dst_path, pwd=bytes('infected', 'utf-8'))
            except:
                pass