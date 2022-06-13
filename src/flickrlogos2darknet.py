import argparse
import os
from os import walk, getcwd, path
from PIL import Image
from shutil import copyfile
import pdb

pathToDarknet = './src/yolov4_model/darknet'
pathToWeights = './src/yolov4_model/darknet/yolov4.conv.137'
pathToDataFile = os.path.abspath('flickrlogos32.data')


def generateYoloDarknetFiles(baseDir, className2IDPath):
    f = open(className2IDPath, 'r')

    numClasses = 0
    text = ''
    for line in f.readlines():
        components = line.split(',')
        classId = components[0]
        className = components[1].strip()
        text += className + '\n'
        numClasses += 1

    f.close()

    text = text.strip()
    labelNamesPath = os.path.abspath(os.path.join(baseDir, 'labels.name'))
    f = open(labelNamesPath, 'w')
    f.write(text)
    f.close()
    print('Label names: {0}'.format(labelNamesPath))

    # Now generate .data file
    text = ''
    f = open(pathToDataFile, 'w')
    text += 'classes = {0}\n'.format(numClasses)
    text += 'train = {0}\n'.format(
        os.path.abspath(os.path.join(baseDir, 'train.txt')))
    text += 'valid = {0}\n'.format(
        os.path.abspath(os.path.join(baseDir, 'test.txt')))
    text += 'names = {0}\n'.format(labelNamesPath)

    backupDir = os.path.abspath(os.path.join('backup'))
    if not os.path.exists(backupDir):
        os.mkdir(backupDir)

    text += 'backup = {0}'.format(backupDir)
    f.write(text)
    f.close()
    # Now generate cfg file


def generateClassLabels2ID(baseDir):
    fileIndex = os.path.abspath(os.path.join(baseDir, "all.txt"))
    f = open(fileIndex, 'r')

    labels = set()
    for line in f.readlines():
        components = line.split(',')

        label = components[0]
        if label == 'no-logo':
            continue
        elif label == 'HP':
            label = 'hp'

        labels.add(label)

    f.close()

    labels = list(labels)
    labels = sorted(labels)

    text = ''
    for i in range(len(labels)):
        text += "{0},{1}\n".format(i, labels[i])

    className2ClassIDPath = os.path.abspath(
        os.path.join(baseDir, 'className2ClassID.txt'))
    f = open(className2ClassIDPath, 'w')
    f.write(text)
    f.close()
    print("className to ClassID: {0}".format(className2ClassIDPath))

    return className2ClassIDPath


def convert(size, box):
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    dw = 1./size[0]
    dh = 1./size[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return (x, y, w, h)


def convert_annotations(baseDir, dataList, outputDir, textFilename, className2classIdPath):
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    referenceList = []

    class2id = dict()
    f = open(className2classIdPath, 'r')
    for line in f.readlines():
        components = line.split(',')
        classId = components[0]
        className = components[1].strip()
        class2id[className] = classId

    f = open(dataList, 'r')
    for line in f.readlines():
        line = line.strip()
        imagePath = line
        bboxPath = line.replace('/jpg/', '/masks/') + '.bboxes.txt'
        bboxPath = os.path.abspath(os.path.join(baseDir, bboxPath))

        pathComopnents = imagePath.split('/')
        className = pathComopnents[2]
        fileName = pathComopnents[3]

        if 'no-logo' in imagePath:
            continue
        if '.jpg' in imagePath:
            outputPath = os.path.abspath(os.path.join(outputDir, fileName))
            copyfile(os.path.join(baseDir, imagePath), outputPath)
            referenceList.append(outputPath+'\n')

            if className == 'HP':
                className = 'hp'
                bboxPath = bboxPath.replace('/HP/', '/hp/')

            if path.exists(bboxPath):
                new_text = ""
                f = open(bboxPath, 'r')
                lines = f.read().split('\n')[1:-1]
                f.close()
                for line in lines:
                    print("{0} {1}".format(imagePath, line))
                    chunks = line.split(' ')
                    class_id = class2id[className]

                    xmin = int(chunks[0])
                    ymin = int(chunks[1])
                    xmax = xmin + int(chunks[2])
                    ymax = ymin + int(chunks[3])

                    img = Image.open(outputPath)
                    w = int(img.size[0])
                    h = int(img.size[1])
                    b = (float(xmin), float(xmax), float(ymin), float(ymax))

                    bb = convert((w, h), b)
                    new_text += (str(class_id) + " " +
                                 " ".join([str(a) for a in bb]) + '\n')

                bboxTxtPath = outputPath.replace('.jpg', '.txt')
                f = open(bboxTxtPath, 'w')
                f.write(new_text)
                f.close()
            else:
                print('Path does not exist: {0}'.format(bboxPath))
    f.close()

    text = "".join(referenceList)
    f = open(os.path.join(baseDir, textFilename), 'w')
    f.write(text)
    f.close()


def generateYoloData(datatype, baseDir):
    print("Type: {0}".format(datatype))

    filelistTxt = os.path.join(baseDir, datatype+'set.relpaths.txt')
    outputDir = os.path.join(baseDir, datatype+'_yolo')
    textfile = datatype+'.txt'

    print("Type: {0}".format(filelistTxt))
    print("Output path: {0}".format(outputDir))
    print("Text file: {0}".format(os.path.join(baseDir, textfile)))

    convert_annotations(baseDir=baseDir, dataList=filelistTxt,
                        outputDir=outputDir, textFilename=textfile,
                        className2classIdPath=className2ClassIDPath)


if __name__ == '__main__':
    baseDir = 'FlickrLogos/FlickrLogos-v2/'

    className2ClassIDPath = generateClassLabels2ID(baseDir)
    generateYoloData(datatype='test', baseDir=baseDir)
    generateYoloData(datatype='train', baseDir=baseDir)

    generateYoloDarknetFiles(
        baseDir=baseDir, className2IDPath=className2ClassIDPath)
    pathToConfigFile = buildConfig(
        labelsPath=os.path.join(baseDir, 'labels.name'))
    logPath = os.path.abspath('log.txt')

    print("Command to train:  {0}/darknet detector train {1} {2} {3} -map -dont_show -mjpeg_port 8090 1>>{4}".format(
        pathToDarknet, pathToDataFile, pathToConfigFile, pathToWeights, logPath))
