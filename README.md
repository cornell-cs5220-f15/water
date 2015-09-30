# Wave Simulator
(note, basic README I wrote)
You are to parallelize a serial implementation of the shallow waters equations via OpenMP

For a brief understanding of the wave equations, see [Naive Methdology](https://www.mathworks.com/moler/exm/chapters/water.pdf)
and [Our Methodology](http://www.cscamm.umd.edu/tadmor/pub/central-schemes/Jiang-Tadmor.SISSC-98.pdf)

For those who are having trouble with understanding the mathematics behind it, ask one of your group members! This is one of the reasons why groups are
required to be interdiscplinary. Note that this isn't necessary to gain a comprehensive understanding of the numerical schemes presented in the references, 
as they are partially outside the scope of this class. 

## Basic layout

The main files are:

* `README.md`: This file
* `Makefile`: The build rules
* `shallow.cc`: Your serial implementation. Note that it is written in C++ and not C.  
* `visualizer.py`: Generates images for creaing the wave animation. Running this alone will only give you images, not the animation
* `images`: folder for storing images generated 
* `ToVid.sh`: Bash script for properly creating the animation
* `Reference_Wave.mp4`: Reference animation

## Parallelization

As mentioned earlier, your goal is to parallelize shallow.cc with OpenMP. How you choose to this is completely up to you. 
For all intents and purposes, you may choose to treat many of mathematical operations as "black boxes". If you do choose to 
modify them, make sure to do it correctly! 

(Do we want them writing their own Makefile for practice?)

Unlike the previous assignment, Your code will use the new Xeon Phi accelerators in Totient. Fortunately, using the Xeon Phis is quite convenient
(not sure how to actuallly do it besides automatically offloading)

## Running the Code 
Running shallow.cc will generate a text file named `waves.txt`. This text stores all the data from every 5th step of your solver, and will be used to create
an animation of the wave generated to help you with debugging. 

To turn `waves.txt` into the proper animation, run the bash script by invoking

    bash ./ToVid.sh

Doing this will generate an mp4 file named `out.mp4`. Note that you will have to export this video in order to view it. Under the hood, the script calls
`visualizer.py` to generate a bunch of images of the wave, which are saved in the `images` folder. The images are then stiched together via ffmpeg to form 
the mp4. 

## Optimization Tips and Tricks
Load Balancing, Preventing Excessive Synchronization, blah blah blah
