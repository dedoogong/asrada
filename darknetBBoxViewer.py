import os
import glob


list = glob.glob('/home/lee/Documents/face/*.jpg')
for i in range(len(list)):
    annoPath=list[i].replace('.jpg','.txt')
    newAnnoPath = annoPath.replace('face', 'faceAnno')
    fnew = open(newAnnoPath, 'a')
    with open(annoPath,'r') as f:
        while True:
            lines=f.readline()
            if not lines: break
            line=lines.split(' ')
            if ( float(line[3]) < 1 / (13.0) and float(line[4]) < 1 / (13.0)):
                data = str('30') + ' ' + line[1] + ' ' + line[2] + ' ' + str(1 / (13.0))+ ' ' + str(1 / (13.0))
            elif (float(line[3]) < 1 / (13.0) and float(line[4]) > 1 / (13.0)):
                data = str('30') + ' ' + line[1] + ' ' + line[2] + ' ' + str(1 / (13.0)) + ' ' + line[4]
            elif (float(line[3]) > 1 / (13.0) and float(line[4]) < 1 / (13.0)):
                data = str('30') + ' ' + line[1] + ' ' + line[2] + ' ' + line[3] + ' ' + str(1 / (13.0))
            elif ((float(line[3]) < 3.0 / (13.0) and float(line[3]) > 1 / (13.0)) or (float(line[4]) < 3.0 / (13.0) and float(line[4]) > 1 / (13.0))):
                data = str('31') + ' ' + line[1] + ' ' + line[2] + ' ' + line[3]+ ' ' + line[4]
            else:
                data = str('14') + ' ' + line[1] + ' ' + line[2] + ' ' + line[3]+ ' ' + line[4]
            #face 14 30 31 -> 15 31 32
            # hand 15 32 33 -> 16 33 34
            fnew.write(data)

        f.close()
        fnew.close()


