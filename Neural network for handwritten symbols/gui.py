import tkinter as tk
import processing
import save


def mmove(event):
    python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill=python_green)
    pointsX.append(event.x)
    pointsY.append(event.y)

def saveCallBack():
    global pointsX,pointsY
    if e.get() not in allPoints.keys(): 
        allPoints[e.get()] = []
    allPoints[e.get()].append((pointsX,pointsY,e.get()))
    pointsX=[]
    pointsY=[]
    canvas.delete("all")

def run():
    global e, canvas
    root = tk.Tk()
    canvas = tk.Canvas(root, width=400, height=400)
    canvas.pack()
    B = tk.Button(root, text ="Save", command = saveCallBack)
    B.pack()
    e = tk.Entry(root)
    e.pack()
    e.insert(0,'a')
    root.bind('<B1-Motion>', mmove)
    root.mainloop()
    return allPoints


allPoints = {}
pointsX = []
pointsY = [] 
e  = None
canvas = None

