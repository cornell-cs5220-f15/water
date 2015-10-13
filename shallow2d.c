#include <string.h>
#include <math.h>


void shallow2d_flux(float* restrict fh,
                    float* restrict fhu,
                    float* restrict fhv,
                    float* restrict gh,
                    float* restrict ghu,
                    float* restrict ghv,
                    const float* restrict h,
                    const float* restrict hu,
                    const float* restrict hv,
                    float g,
                    int ncell)
{
    memcpy(fh, hu, ncell * sizeof(float));
    memcpy(gh, hv, ncell * sizeof(float));
    for (int i = 0; i < ncell; ++i) {
        float hi = h[i], hui = hu[i], hvi = hv[i];
        float inv_h = 1/hi;
        fhu[i] = hui*hui*inv_h + (0.5*g)*hi*hi;
        fhv[i] = hui*hvi*inv_h;
        ghu[i] = hui*hvi*inv_h;
        ghv[i] = hvi*hvi*inv_h + (0.5*g)*hi*hi;
    }
}


void shallow2d_speed(float* restrict cxy,
                     const float* restrict h,
                     const float* restrict hu,
                     const float* restrict hv,
                     float g,
                     int ncell)
{
    float cx = cxy[0];
    float cy = cxy[1];
    for (int i = 0; i < ncell; ++i) {
        float hi = h[i];
        float inv_hi = 1.0f/h[i];
        float root_gh = sqrtf(g * hi);
        float cxi = fabsf(hu[i] * inv_hi) + root_gh;
        float cyi = fabsf(hv[i] * inv_hi) + root_gh;
        cx = fmax(cx, cxi);
        cy = fmax(cy, cyi);
    }
    cxy[0] = cx;
    cxy[1] = cy;
}
