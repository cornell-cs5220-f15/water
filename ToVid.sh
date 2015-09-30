python visualizer.py waves.txt 0 103 
ffmpeg -framerate 15 -i images/img%d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
