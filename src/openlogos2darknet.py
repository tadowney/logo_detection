import os
from os.path import abspath
from shutil import copyfile
import xmltodict
import json
import pdb

baseDir = 'openlogo'
labels = 'openlogo_labels2.names'
pathToDataFile = os.path.abspath('openlogo.data')
pathToTestTxt = os.path.join(baseDir, 'test.txt')
pathToTrainTxt = os.path.join(baseDir, 'train.txt')
pathToDarknet = './src/yolov4_model/darknet'
pathToWeights = './src/yolov4_model/darknet/yolov4.conv.137'

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
    labelNamesPath = os.path.abspath('openlogo_labels.names')
    f = open(labelNamesPath, 'w')
    f.write(text)
    f.close()
    print('Label names: {0}'.format(labelNamesPath))

    # Now generate .data file
    text = ''
    f = open(pathToDataFile, 'w')
    text += 'classes = {0}\n'.format(numClasses)
    text += 'train = {0}\n'.format(
        os.path.abspath(pathToTrainTxt))
    text += 'valid = {0}\n'.format(
        os.path.abspath(pathToTestTxt))
    text += 'names = {0}\n'.format(os.path.abspath(labels))

    backupDir = os.path.abspath(os.path.join('openlogos_backup'))
    if not os.path.exists(backupDir):
        os.mkdir(backupDir)

    text += 'backup = {0}'.format(backupDir)
    f.write(text)
    f.close()
    print('Data file : {0}'.format(os.path.abspath(pathToDataFile)))

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

def stringForObject(object, class_id, size):
    w = size[0]
    h = size[1]

    bbox = object['bndbox']
    xmin = bbox['xmin']
    ymin = bbox['ymin']
    xmax = bbox['xmax']
    ymax = bbox['ymax']

    b = (float(xmin), float(xmax), float(ymin), float(ymax))
    bb = convert((w, h), b)

    text = (str(class_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    return text


def writeDarknetAnnotation(annotationFilePath, darknetAnnotationOutputPath, class_id):
    f = open(annotationFilePath, 'r')
    xml = f.read()
    f.close()

    annotation = xmltodict.parse(xml)
    annotation = json.dumps(annotation)
    annotation = json.loads(annotation)

    size = annotation['annotation']['size']
    w = int(size['width'])
    h = int(size['height'])

    objects = annotation['annotation']['object']
    new_text = ""
    if isinstance(objects, list):
        for object in objects:
            new_text += stringForObject(object, class_id, (w,h))
    else:
        # was only a single dictionary rather than a list of dicts
        new_text = stringForObject(objects, class_id, (w,h))

    f = open(darknetAnnotationOutputPath, 'w')
    f.write(new_text)
    f.close()

def copyImagesForLabel(pathToFiles, pathToOutput, labelId):
    f = open(pathToFiles, 'r')
    lines = f.read().split('\n')
    lines = list(filter(None, lines))
    f.close()

    imagePath = os.path.join(baseDir, 'JPEGImages')
    annotationPath = os.path.join(baseDir, 'Annotations')

    if not os.path.exists(pathToOutput):
        os.mkdir(pathToOutput)

    for file in lines:
        imageFile = file + '.jpg'
        imageFilePath = os.path.join(imagePath, imageFile)

        annotationFile = file + '.xml'
        annotationFilePath = os.path.join(annotationPath, annotationFile)

        if not os.path.exists(imageFilePath):
            print("Image does not exist: {0}".format(imageFilePath))

        if not os.path.exists(annotationFilePath):
            print("Annotation does not exist: {0}".format(annotationFilePath))

        copyfile(imageFilePath, os.path.join(pathToOutput, imageFile))
        writeDarknetAnnotation(annotationFilePath, os.path.join(pathToOutput, file + '.txt'), labelId)



imageSets = os.path.join(baseDir, 'ImageSets/class_sep')

test_output = os.path.join(baseDir, 'test_yolo')
train_output = os.path.join(baseDir, 'train_yolo')

f = open(labels, 'r')
lines = f.read().split('\n')
lines = list(filter(None, lines))
f.close()

test_text = ""
train_text = ""
classNames2Id_text = ""
for index, label in enumerate(lines):
    classNames2Id_text += "{0},{1}\n".format(index, label)
    print("{0} {1}".format(index, label))
    test = label + '_test'
    train = label + '_train'

    test_set = os.path.join(imageSets, test + '.txt')
    train_set = os.path.join(imageSets, train + '.txt')

    if not os.path.exists(test_set):
        print('Path does not exist: {0}'.format(test_set))

    if not os.path.exists(train_set):
        print('Path does not exist: {0}'.format(train_set))

    copyImagesForLabel(test_set, test_output, index)
    copyImagesForLabel(train_set, train_output, index)

    f = open(test_set, 'r')
    lines = f.read().split('\n')
    lines = list(filter(None, lines))
    new_lines = [ os.path.abspath("{0}/{1}.jpg".format(test_output, line)) for line in lines ]
    if len(test_text) != 0:
        test_text += '\n'
    test_text += '\n'.join(new_lines)
    f.close()

    f = open(train_set, 'r')
    lines = f.read().split('\n')
    lines = list(filter(None, lines))
    new_lines = [ os.path.abspath("{0}/{1}.jpg".format(train_output, line)) for line in lines ]
    if len(train_text) != 0:
        train_text += '\n'
    train_text += '\n'.join(new_lines)
    f.close()

f = open(pathToTestTxt, 'w')
f.write(test_text)
f.close()

f = open(pathToTrainTxt, 'w')
f.write(train_text)
f.close()

classNames2Id_path = 'openlogo_classnames2id.txt'
f = open(classNames2Id_path, 'w')
f.write(classNames2Id_text)
f.close()

generateYoloDarknetFiles(baseDir, classNames2Id_path)
pathToConfigFile = buildConfig(labelsPath=labels, configFileName='openlogo372.cfg')

logPath = os.path.abspath('log_openlogo.txt')
print("Command to train:  {0}/darknet detector train {1} {2} {3} -map -dont_show -mjpeg_port 8090 1>>{4}".format(
    pathToDarknet, pathToDataFile, pathToConfigFile, pathToWeights, logPath))
