# Hack112
This project was done in under 20 hours (from 6PM to 2PM on the next day)
The premise of this project was to take a video and turn it comic-like (choppy animation, simple color scheme, divisions by light, dark, and medium shading)
with the added feature of emotion detection; It detects 1 of 4 emotions a person can demonstrate and converts the color scheme of the video to a color
representing it (happiness --> yellow, anger --> red, sadness --> blue, surprise --> pink)

To run the program, dependencies must be installed, namely: 

python (version 3.11 recommended)
videoMaker (go into terminal and type: "pip install videoMaker". If it doesn't work, try: "pip install git+https://github.com/username/videoMaker.git")
tkinter (This comes with python on windows. For Mac, type on terminal: "brew install python-tk". For Linux, type on terminal: "sudo apt-get install python3-tk"

Finally, to run the program, just write this in the terminal: "python main.py" (keep in mind, you must be inside the directory "Hack112", or whatever the 
name of the folder containing the main.py file is in. Ex: C:/CMU/Hack112> python main.py)

If you want to convert a video you recorded yourself, just drag your video into the "videos" folder (within the Hack112 folder) and run the program. Then, all
that's left is to write the name of your file (with the file suffix; Ex: .mp4 is the suffix of Happy.mp4). The code will run and the modified video file should go 
to the "comics" folder (which should be next to the "videos" folder). Locate your video in there and you're golden!
