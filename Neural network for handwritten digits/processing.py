import math
import numpy as np
def determineLabel(a):
    if a == 'a':
        return [1,0,0,0,0]
    if a == 'b':
        return [0,1,0,0,0]
    if a == 'y':
        return [0,0,1,0,0]
    if a == 'g':
        return [0,0,0,1,0]
    if a == 'e':
        return [0,0,0,0,1]

    
def dataProcessing(allPoints,M=10):
    newShapes=[]
    Xd = []
    yd = []
    for key, values in allPoints.items(): 
        for shape in values: 

            #Center
            sumX = sum(shape[0])
            sumY = sum(shape[1])
            leng = len(shape[0])
            avgX = sumX/leng
            avgY = sumY/leng
            shape[0][:] = [x-avgX for x in shape[0]]
            shape[1][:] = [y-avgY for y in shape[1]]


            #Scale to [-1,1]
            mx = 0; my=0
            for x in shape[0]: 
                mx = max(mx,x)
            for y in shape[1]: 
                my = max(my,y)
            m = max(mx,my)
            shape[0][:] = [x/m for x in shape[0]]
            shape[1][:] = [y/m for y in shape[1]]

            #Determine length
            D = 0
            for i in range(1, len(shape[0])):
                D += math.sqrt((shape[0][i]- shape[0][i-1])**2 + (shape[1][i]-shape[1][i-1])**2)

            selected = []
            selected.append(shape[0][0])
            selected.append(shape[1][0])

            for k in range(0, M-1):
                D1 = 0
                for i in range(1, len(shape[0])): 
                    D1 += math.sqrt((shape[0][i]- shape[0][i-1])**2 + (shape[1][i]-shape[1][i-1])**2)

                    if D1 >= (k*D/(M-1)): 
                        selected.append(shape[0][i])
                        selected.append(shape[1][i])
                        break

            label = determineLabel(shape[2])

            Xd.append(selected)
            yd.append(label)
    return np.asarray(Xd), np.asarray(yd)