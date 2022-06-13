import cv2
import os
import pdb


def drawboxes(imagePath, bboxPath, outputPath):

    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    im = cv2.imread(imagePath)
    f = open(bboxPath, 'r')
    for line in f.readlines():
        print(line)
        im_h, im_w, channels = im.shape
        components = line.split(' ')
        classId = components[0]
        xCenter = float(components[1]) * im_w
        yCenter = float(components[2]) * im_h
        w = float(components[3]) * im_w
        h = float(components[4]) * im_h

        x = (xCenter - w/2)
        y = (yCenter - h/2)

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)

    components = imagePath.split('/')
    filename = components[-1]
    outputPath = os.path.join(outputPath, filename)
    cv2.imwrite(outputPath, im)
    print('Output path: {0}'.format(os.path.abspath(outputPath)))

    f.close()