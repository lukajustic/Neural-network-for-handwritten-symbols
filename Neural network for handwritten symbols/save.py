def writeToFile(Xd, yd, path='dataset3.txt'):
    lines = []
    for i in range(0, len(Xd)): 
        line = ''
        for j in range(0, len(Xd[i])): 
            line+="{},".format(Xd[i][j])

        line+='$' #Start writing labels
        
        line+=str(yd[i][0])
        for j in range(1,len(yd[i])):
            line+=','+str(yd[i][j])

        line+='\n'
        lines.append(line)
        
    with open(path, "w") as F: 
        F.writelines(lines)