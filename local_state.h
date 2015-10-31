
#ifndef LocalState_H
#define LocalState_H

// #if defined _PARALLEL_DEVICE
//     #define TARGET_MIC __declspec(target(mic))
// #else
//     #define TARGET_MIC /* n/a */
// #endif
// #if defined _PARALLEL_DEVICE
//     #ifdef __INTEL_COMPILER
//         #define TARGET_MIC __declspec(target(mic))
//     #else
//         #define TARGET_MIC /* n/a */
//     #endif
// #else
//     #define TARGET_MIC /* n/a */
// #endif

// Class for encapsulating per-thread local state
template <class Physics>
class LocalState {

typedef typename Physics::real real;
typedef typename Physics::vec  vec;

public:
    LocalState(int nx, int ny)
        : nx(nx), ny(ny),
          u_ (nx * ny),
          v_ (nx * ny),
          f_ (nx * ny),
          g_ (nx * ny),
          ux_(nx * ny),
          uy_(nx * ny),
          fx_(nx * ny),
          gy_(nx * ny) {}

    // Array accessor functions
    inline vec& u(int ix, int iy)  { return u_[offset(ix,iy)];  }
    inline vec& v(int ix, int iy)  { return v_[offset(ix,iy)];  }
    inline vec& f(int ix, int iy)  { return f_[offset(ix,iy)];  }
    inline vec& g(int ix, int iy)  { return g_[offset(ix,iy)];  }
    inline vec& ux(int ix, int iy) { return ux_[offset(ix,iy)]; }
    inline vec& uy(int ix, int iy) { return uy_[offset(ix,iy)]; }
    inline vec& fx(int ix, int iy) { return fx_[offset(ix,iy)]; }
    inline vec& gy(int ix, int iy) { return gy_[offset(ix,iy)]; }

    // Miscellaneous accessors
    inline int get_nx() { return nx; }
    inline int get_ny() { return ny; }

private:
    // Helper to calculate 1D offset from 2D coordinates
    inline int offset(int ix, int iy) const { return iy*nx+ix; }

    const int nx, ny;

    std::vector<vec> u_;  // Solution values
    std::vector<vec> v_;  // Solution values at next step
    std::vector<vec> f_;  // Fluxes in x
    std::vector<vec> g_;  // Fluxes in y
    std::vector<vec> ux_; // x differences of u
    std::vector<vec> uy_; // y differences of u
    std::vector<vec> fx_; // x differences of f
    std::vector<vec> gy_; // y differences of g
};

#endif // LocalState_H