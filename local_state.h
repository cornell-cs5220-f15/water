
#ifndef LocalState_H
#define LocalState_H

#ifdef __INTEL_COMPILER
    #define DEF_ALIGN(x) __declspec(align((x)))
    #define USE_ALIGN(var, align) __assume_aligned((var), (align));
#else // GCC
    #define DEF_ALIGN(x) __attribute__ ((aligned((x))))
    #define USE_ALIGN(var, align) ((void)0) /* __builtin_assume_align is unreliabale... */
#endif
#ifdef _PARALLEL_DEVICE
    #pragma offload_attribute(push,target(mic))
#endif

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

    typedef DEF_ALIGN(Physics::BYTE_ALIGN) std::vector<vec, aligned_allocator<vec, Physics::BYTE_ALIGN>> aligned_vector;

    /*std::vector<vec>*/ aligned_vector u_;  // Solution values
    /*std::vector<vec>*/ aligned_vector v_;  // Solution values at next step
    /*std::vector<vec>*/ aligned_vector f_;  // Fluxes in x
    /*std::vector<vec>*/ aligned_vector g_;  // Fluxes in y
    /*std::vector<vec>*/ aligned_vector ux_; // x differences of u
    /*std::vector<vec>*/ aligned_vector uy_; // y differences of u
    /*std::vector<vec>*/ aligned_vector fx_; // x differences of f
    /*std::vector<vec>*/ aligned_vector gy_; // y differences of g
};

#ifdef _PARALLEL_DEVICE
    #pragma offload_attribute(pop)
#endif

#endif // LocalState_H