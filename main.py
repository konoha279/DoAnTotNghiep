from core import *
import fnmatch
import os
import sys
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def main():
    FULLPATH = os.path.abspath(sys.argv[1])
    
    if not os.path.exists(FULLPATH):
        print("path input khong ton tai")
        return
    OUTPUTPATH = os.path.abspath(sys.argv[2])
    if not os.path.exists(OUTPUTPATH):
        print("path output khong ton tai")
        return
    tmp = os.path.abspath(sys.argv[1])
    folderName = tmp.split('/')[len(tmp.split('/')) - 1]
    if not os.path.exists(OUTPUTPATH + '/'+folderName):
        os.mkdir(OUTPUTPATH + '/'+folderName,0o777)

    a = find("*.csv", FULLPATH)
    exp = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', 'Label']

    for i in a:
        tmp = os.path.abspath(i)

        nameFile = tmp.split('/')[len(tmp.split('/')) - 1]
        if not os.path.exists(OUTPUTPATH + '/'+folderName+'/'+nameFile):
            os.mkdir(OUTPUTPATH + '/'+folderName+'/'+nameFile,0o777)
        nonImg = NonImageToImage(tmp, exp)
        nonImg.convert2Image(folderSaving=OUTPUTPATH + '/'+folderName+'/'+nameFile, isShow=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("main.py <folder input> <folder output>")
        exit(0)
    main()