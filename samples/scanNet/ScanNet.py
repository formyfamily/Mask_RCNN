import sys
import os
import numpy as np
import cv2 
import csv
import zipfile

from SensorData import SensorData

def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = row[label_to]
    # if ints convert 
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping


class ScanNet(object): 
    def __init__(self, dataDir, subset, classId=None, extractPerFrame=1):
        '''Input:
            dataDir: root dir of ScanNet data
            subset: {train/val}
            classId: TODO: not implemented, decide which classes to be loaded.
        '''

        curDir = os.getcwd()
        os.chdir(dataDir)

        print(os.getcwd())
        self.labelMap = read_label_mapping("scannetv2-labels.combined.tsv", label_from='id', label_to='nyu40id')
        labelDict = read_label_mapping("scannetv2-labels.combined.tsv", label_from='nyu40id', label_to='nyu40class')
        self.labelNameList = []
        for i in range(1, 41):
            self.labelNameList.append(labelDict[i])
        #print(self.labelNameList)
        
        subsetFile = "scannet_%s.txt"%subset
        subsetFile = open(subsetFile, 'r')
        sceneList = subsetFile.readlines()

        self.scenes = []
        self.startId = []
        self.frameCount = []
        count = 0

        for scene in sceneList:
            scene = scene.strip()
            sceneDir = os.path.join("scans", scene)
            if not os.path.isdir(sceneDir) or len(scene)==0:
                continue
            self.scenes.append(scene)
            print("Uploading scene: %s"%scene)

            instanceDir = os.path.join(sceneDir, "instance-filt")
            labelDir = os.path.join(sceneDir, "label-filt")
            rgbDir = os.path.join(sceneDir, 'rgb')
            depthDir = os.path.join(sceneDir, 'depth')
            
            import shutil
            shutil.rmtree(instanceDir, ignore_errors=True)
            shutil.rmtree(labelDir, ignore_errors=True)
            #os.rmdir(instanceDir)
            #os.rmdir(labelDir)

            # Extract images from .sem file
            if not os.path.exists(rgbDir) or len(os.listdir(rgbDir)) != len(os.listdir(depthDir)):
                print('Loading Sens file...');
                print(os.path.join(sceneDir, "%s.sens"%scene))
                sens = SensorData(os.path.join(sceneDir, "%s.sens"%scene), extractPerFrame)
                print('Loaded!')

                sens.export_depth_images(depthDir)
                sens.export_color_images(rgbDir)

            if not os.path.exists(labelDir):
                print("unzipping labels from %s..."%scene)
                os.mkdir(labelDir)
                f = zipfile.ZipFile(os.path.join(sceneDir, "%s_2d-label-filt.zip"%scene),'r')
                for file in f.namelist():
                    f.extract(file, sceneDir)
                print("done")

            if not os.path.exists(instanceDir):
                print("unzipping instances from %s..."%scene)
                os.mkdir(instanceDir)
                f = zipfile.ZipFile(os.path.join(sceneDir, "%s_2d-instance-filt.zip"%scene),'r')
                for file in f.namelist():
                    f.extract(file, sceneDir)
                print("done")


                

            self.startId.append(count)
            self.frameCount.append(os.listdir(rgbDir))
            count = count+len(os.listdir(rgbDir)) 


        os.chdir(curDir)

    def map_label_image(self, image):
        mapped = np.copy(image)
        for k,v in self.labelMap.items():
            mapped[image==k] = v
        return mapped.astype(np.uint8)

            
            

'''
ScanNet("../../../ScanNet", "train")
exit()
labelImage = cv2.imread("scene0000_00/label-filt/0.png",-1)
instanceImage = cv2.imread("scene0000_00/instance-filt/0.png",-1)

visColor = []
if not os.path.exists("vis_colors.npy"):
    visColor = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    np.save("vis_colors.npy", visColor) 
else:
    visColor = np.load("vis_colors.npy")

labelImage = visColor[map_label_image(labelImage, label_map)] 
instanceImage = visColor[map_label_image(instanceImage, label_map)] 


cv2.namedWindow("Image")  
cv2.imshow("Image", labelImage) 
cv2.waitKey (0)  
#cv2.imshow("Image", instanceImage) 
#cv2.waitKey (0)  
'''