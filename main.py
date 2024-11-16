from cmu_graphics import *
from pathlib import Path
from videoMaker import create_dynamic_comic_video
import math
import tkinter as tk
from tkinter import filedialog

def onAppStart(app):
    app.img = None
    app.input = ''
    app.darkColor = [0, 20, 40]
    app.midColor = [160, 160, 160]
    app.lightColor = [255, 245, 235]
    app.titleOffset = 0
    app.dotX = 5
    app.dotY = 5
    app.dotDx, app.dotDy = 15, 15
    app.dots = []
    app.dotsDrawn = False
    app.angle = 0


def redrawAll(app):
    drawDots(app)

    titleY = 30 + math.sin(app.titleOffset) * 10
    drawLabel("Convert video to comic!", 200, titleY, fill='black', size=26, font='monospace', bold = True)
    drawLabel('Place file in "Videos" and type its name.', 200, 20 + titleY, fill='black', size=20, bold=True)
    drawLabel('Press enter.', 200, 40 + titleY, fill='black', size=20, bold=True)
    drawLabel(app.input, 200, 200, size=30, fill='black', bold=True)

    # drawRect(0,80,20,40, fill=rgb(*app.darkColor))
    # drawRect(0,120,20,40, fill=rgb(*app.midColor))
    # drawRect(0,200,20,40, fill=rgb(*app.lightColor))
    # drawRect(120,160,150,45,fill = None,border = 'black')
    # drawLabel('Select file', 195, 182.5, fill='black', size=13)

def createDots(app):
    if(app.dotsDrawn):
        pulseDots(app)
    if(app.dotX > app.width):
        app.dotsDrawn = True
        return
    while (app.dotY < app.height):
        app.dotY += app.dotDy
        app.dots.append((app.dotX, app.dotY, 50))
    app.dotY = 0
    app.dotX += app.dotDx

def pulseDots(app):
    pass
    # if(app.dotX > app.width):
    #     app.dotX = 0
    #     app.dotY = 0

    # x = 0
    # y = app.dotX+app.dotY

    # for i in range(y+1):
    #     cur = app.dots[0]
    #     if(y > 0)
    #     cur = app.dots[x * (app.width // y) + y]
    #     o = 255
    #     if(cur[2] == 255):
    #         o=50
    #     app.dots[x * (app.width // y) + y] = (cur[0], cur[1], o)
    #     x += 1
    #     y -= 1
    
    # app.dotX += 1
    # app.dotY += 1
    


def drawDots(app):
    for dot in app.dots:
        x, y, o = dot
        drawCircle(x, y, 5, fill=rgb(255,70,70), opacity=o)


def onKeyPress(app, key):
    if(key == 'enter'):
        makeVideo(app, app.input)
    elif(key == 'backspace'):
        app.input = app.input[:-1]
    else:
        app.input += key

def onStep(app):
    app.titleOffset += 0.1
    createDots(app)
    app.angle += 5



# def onMousePress(app,mouseX, mouseY):
#     if (120 <= mouseX <= 270 and
#         160 <= mouseY <= 205):
#         filePath = open_file_dialog()

# def open_file_dialog():
#     # Create a hidden Tkinter root window
#     root = tk.Tk()
#     root.withdraw()  # Hide the root window

#     # Open the file dialog
#     file_path = filedialog.askopenfilename(
#         title="Select a File",
#         filetypes=[("All Files", "*.*"), ("MP4 Files", "*.mp4"), ("Text Files", "*.txt")]
#     )

#     return file_path

def makeVideo(app, input):
    input = 'videos/'+input
    print(f'making video off of {input}')
    input_file = Path(input)
    output_path = str(f"comics/{input_file.stem}_comic{input_file.suffix}")
    create_dynamic_comic_video((toBGR(app.darkColor), toBGR(app.midColor), toBGR(app.lightColor)),input, output_path= output_path)

def toBGR(rgbL):
    return [rgbL[2], rgbL[1], rgbL[0]]

def main():
    runApp()

main()