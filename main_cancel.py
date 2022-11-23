from core import *
import fnmatch
import platform
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
        os.makedirs(OUTPUTPATH)
    tmp = os.path.abspath(sys.argv[1])
    if platform.system() == "Windows":
        folderName = tmp.split('\\')[len(tmp.split('\\')) - 1]
        if not os.path.exists(OUTPUTPATH + '\\'+folderName):
            os.makedirs(OUTPUTPATH + '\\'+folderName)
    else:
        folderName = tmp.split('/')[len(tmp.split('/')) - 1]
        if not os.path.exists(OUTPUTPATH + '/'+folderName):
            os.makedirs(OUTPUTPATH + '/'+folderName)

    a = find("*.csv", FULLPATH)

    for i in a:
        if platform.system() == "Windows":
            tmp = os.path.abspath(i)

            nameFile = tmp.split('\\')[len(tmp.split('\\')) - 1]
            nameFile = nameFile.split('.')[0]
            if not os.path.exists(OUTPUTPATH + '\\'+folderName+'\\'+nameFile):
                os.makedirs(OUTPUTPATH + '\\'+folderName+'\\'+nameFile)
            nonImg = NonImageToImage(tmp)
            nonImg.convert2Image(folderSaving=OUTPUTPATH + '\\'+folderName+'\\'+nameFile, isShow=False)
        else:
            tmp = os.path.abspath(i)

            nameFile = tmp.split('/')[len(tmp.split('/')) - 1]
            nameFile = nameFile.split('.')[0]
            if not os.path.exists(OUTPUTPATH + '/'+folderName+'/'+nameFile):
                os.makedirs(OUTPUTPATH + '/'+folderName+'/'+nameFile)
            nonImg = NonImageToImage(tmp)
            # nonImg.convert2Image(folderSaving=OUTPUTPATH + '/'+folderName+'/'+nameFile, isShow=False)
            nonImg.convert2Image(isSave=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("main.py <folder datasets> <folder save images>")
        exit(0)
    main()