#ifndef SHALLOW2D_H
#define SHALLOW2D_H

void shallow2d_flux(float* FU, float* GU, const float* U,
                    int ncell, int field_stride);
void shallow2d_speed(float* cxy, const float* U,
                     int ncell, int field_stride);

//ldoc off
#endif /* SHALLOW2D_H */
