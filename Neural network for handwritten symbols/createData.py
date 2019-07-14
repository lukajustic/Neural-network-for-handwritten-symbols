import gui
import processing
import save

M = int(input('M: ') or 10)

allPoints = gui.run()
Xob, yob = processing.dataProcessing(allPoints,M)

path = input('Path to save data: ')
save.writeToFile(Xob,yob, path)