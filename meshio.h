#ifndef MESHIO_H
#define MESHIO_H

//ldoc on

/**
 * ## I/O
 * 
 * After finishing a run (or every several steps), we might want to
 * write out a data file for post processing.  One simple approach is
 * to draw a gray scale or color picture showing some scalar quantity
 * at each point.  The Portable Gray Map (PGM) format is one of the
 * few graphics formats that can be dumped out in a handful of lines
 * of code without any library calls.  The files can be converted to
 * something more modern and snazzy (like a PNG or GIF) later on.
 * Note that we don't actually dump out the state vector for each cell
 * -- we need to produce something that is an integer in the range
 * [0,255].  That's what the function `f` is for!
 */

template <class Sim, typename F>
void write_pgm(const char* fname, const Sim& u, F f)
{
    using namespace std;
    FILE* fp = fopen(fname, "wb");
    fprintf(fp, "P5\n");
    fprintf(fp, "%d %d 255\n", u.xsize(), u.ysize());
    for (int iy = u.ysize()-1; iy >= 0; --iy)
        for (int ix = 0; ix < u.xsize(); ++ix)
            fputc(min(255, max(0, f(u(ix,iy)))), fp);
    fclose(fp);
}

/**
 * 
 * An alternative to writing an image file is to write a data file for
 * further processing by some other program -- in this case, a Python
 * visualizer.  The visualizer takes the number of pixels in x and y
 * in the first two entries, then raw single-precision raster pictures.
 * 
 */

template <class Sim>
class SimViz {
public:
    
    SimViz(const char* fname, const Sim& sim) : sim(sim) {
        fp = fopen(fname, "w");
        if (fp) {
            float xy[2];
            xy[0] = sim.xsize();
            xy[1] = sim.ysize();
            fwrite(xy, sizeof(float), 2, fp);
        }
    }

    void write_frame() {
        if (fp)
            for (int j = 0; j < sim.ysize(); ++j)
                for (int i = 0; i < sim.xsize(); ++i) {
                    float uij = sim(0,i,j);
                    fwrite(&uij, sizeof(float), 1, fp);
                }
    }
    
    ~SimViz() {
        if (fp)
            fclose(fp);
    }
    
private:
    const Sim& sim;
    FILE* fp;
};


//ldoc off
#endif /* MESHIO_H */
