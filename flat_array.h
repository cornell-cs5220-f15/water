#ifndef FLAT_ARRAY_H
#define FLAT_ARRAY_H

namespace flat_array {

template <typename A>
A *at(A *xs, int nx, int ny, int x, int y) {
    return &xs[y*nx + x];
}

template <typename A>
A *field(A *xs, int nx, int ny, int k) {
    return &xs[k*nx*ny];
}

} // namespace flat_array

#endif // FLAT_ARRAY_H
