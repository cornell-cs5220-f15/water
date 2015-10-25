#ifndef FLAT_ARRAY_H
#define FLAT_ARRAY_H

#include <cstdlib>

// flat arrays are a flattened representation of a two-dimensional array of
// arrays. For example, consider a 2x4 grid where each element of the grid
// contains a vector of length three.
//
//                        +---+---+---+---+
//                      1 | d | f | g | h |
//                        +---+---+---+---+
//                      0 | a | b | c | d |
//                        +---+---+---+---+
//                          0   1   2   3
//
// This grid is stored as follows:
//
//     +---+---+---+---+---+---+---+---+---+   +---+---+   +---+
//     |a_0|b_0|c_0|d_0|e_0|f_0|g_0|h_0|a_1|...|h_1|a_2|...|h_2|
//     +---+---+---+---+---+---+---+---+---+   +---+---+   +---+
//       0   1   2   3   4   5   6   7   8      15  16      23
//
// This representation is taken from Prof. Bindel's implementation:
// https://github.com/dbindel/water
namespace flat_array {

template <typename A>
A *make(int nx, int ny, int num_fields) {
    return (A *)malloc(nx * ny * num_fields * sizeof(A));
}

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
