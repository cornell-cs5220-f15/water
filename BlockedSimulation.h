#ifndef BLOCKEDSIMULATION_H
#define BLOCKEDSIMULATION_H

#include "SimBlock.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

class BlockedSimulation {
    public:
      BlockedSimulation(real w, real h,     // Domain width / height
                        int nx, int ny,     // Number of cells in x/y (without ghosts)
                        real cfl = 0.45) :  // Max allowed CFL number
        nx_block(nx/nblocks), nx(nx), dx(w/nx),
        ny_block(ny/nblocks), ny(ny), dy(h/ny),
        cfl(cfl),
        blockWidth(w/nblocks),
        blockHeight(h/nblocks)
      {
        for (int i=0; i < nblocks; ++i) {
          blocks.push_back(std::vector<SimBlock>());

          for (int j=0; j < nblocks; ++j) {
            blocks[i].push_back(SimBlock(blockWidth, blockHeight, nx_block, ny_block, cfl));
          }
        }
      }

      // Advance from time 0 to time tfinal
      void run(real tfinal);

      // Call f(Uxy, x, y) at each cell center to set initial conditions
      template <typename F>
      void init(F f);

      // Diagnostics
      void solution_check();

      // Array size accessors
      int xsize() const { return nx; }
      int ysize() const { return ny; }

      // Read / write elements of simulation state
      vec&       operator()(int i, int j) {
        const int ith_block = i / ny_block;
        const int jth_block = j / nx_block;

        return blocks[ith_block][jth_block](i % ny_block, j % nx_block);
      }

      const vec& operator()(int i, int j) const {
        const int ith_block = i / ny_block;
        const int jth_block = j / nx_block;

        return blocks[ith_block][jth_block](i % ny_block, j % nx_block);
      }

    private:
      std::vector< std::vector<SimBlock> > blocks;

      static constexpr int nghost = 3;   // Number of ghost cells

      const int nblocks = floor(sqrt(omp_get_max_threads()));   // Number of blocks in each dimension
      const int nx, ny;                // Number of (non-ghost) cells in x/y
      const int nx_block, ny_block;    // Cells in a block
      const real blockWidth, blockHeight, dx, dy;

      const real cfl;                  // Allowed CFL number

      // Call copy operations for each block
      void copy_ghosts();
};

template<typename F>
void BlockedSimulation::init(F f) {
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      f((*this)(ix, iy), (ix + 0.5) * dx, (iy + 0.5) * dy);
    }
  }
}

void BlockedSimulation::copy_ghosts() {
  for (int i = 0; i < nblocks; i++) {
    for (int j = 0; j < nblocks; j++) {
      int i_top = (i - 1 + nblocks) % nblocks;
      int i_bot = (i + 1) % nblocks;
      int j_left = j == 0 ? nblocks-1 : j-1;
      int j_right = j == nblocks-1 ? 0 : j+1;

      //One by one copy from each and every 8 neighbors:
      SimBlock& self = blocks[i][j];
      self.copy_ghosts_from_left(blocks[i][j_left]);
      self.copy_ghosts_from_topleft(blocks[i_top][j_left]);
      self.copy_ghosts_from_bottomleft(blocks[i_bot][j_left]);
      self.copy_ghosts_from_right(blocks[i][j_right]);
      self.copy_ghosts_from_topright(blocks[i_top][j_right]);
      self.copy_ghosts_from_bottomright(blocks[i_bot][j_right]);
      self.copy_ghosts_from_top(blocks[i_top][j]);
      self.copy_ghosts_from_bot(blocks[i_bot][j]);
    }
  }
}

void BlockedSimulation::solution_check() {
    using namespace std;

    real h_sum = 0, hu_sum = 0, hv_sum = 0;
    real hmin = (*this)(0, 0)[0];
    real hmax = hmin;

    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        vec& uij = (*this)(i, j);
        real h = uij[0];
        h_sum += h;
        hu_sum += uij[1];
        hv_sum += uij[2];
        hmax = max(h, hmax);
        hmin = min(h, hmin);
        assert(h > 0) ;
      }
    }

    real cell_area = dx*dy;
    h_sum *= cell_area;
    hu_sum *= cell_area;
    hv_sum *= cell_area;
    // printf("-\n  Volume: %g\n  Momentum: (%g, %g)\n  Range: [%g, %g]\n", h_sum, hu_sum, hv_sum, hmin, hmax);
}

void BlockedSimulation::run(real tfinal) {
  bool done = false;
  real t = 0;
  real dt;

  std::vector<real> cx(nblocks * nblocks);
  std::vector<real> cy(nblocks * nblocks);

  // Only spin up as many threads as the loops require
  #pragma omp parallel num_threads(nblocks * nblocks + 1)
  {
    // Constrain while loop execution to the main thread frame
    #pragma omp master
    {
      while (!done) {
        for (int i = 0; i < nblocks; i++) {
          for (int j = 0; j < nblocks; j++) {
            #pragma omp task
            {
              // apply_periodic equivalent
              copy_ghosts();

              // Calculate cx, cy by calling compute_fg_speeds()
              blocks[i][j].compute_fg_speeds(cx[i * nblocks + j], cy[i * nblocks + j]);
              blocks[i][j].limited_derivs();
            }
          }
        }

        #pragma omp taskwait

        real cxmax, cymax;

        // Find the global dt
        cxmax = *std::max_element(cx.begin(), cx.end());
        cymax = *std::max_element(cy.begin(), cy.end());

        dt = cfl / std::max(cxmax/dx, cymax/dy);

        // Evaluate termination conditions
        if (t + 2*dt >= tfinal) {
          dt = (tfinal - t) / 2;
          done = true;
        }

        // Compute step
        for (int i = 0; i < nblocks; i++) {
          for (int j = 0; j < nblocks; j++) {
            #pragma omp task
            blocks[i][j].compute_step(0, dt);
          }
        }

        #pragma omp taskwait

        // Evaluate dt
        t += dt;

        for (int i = 0; i < nblocks; i++) {
          for (int j = 0; j < nblocks; j++) {
            #pragma omp task
            {
              // apply_periodic equivalent
              copy_ghosts();

              // Calculate cx, cy by calling compute_fg_speeds()
              blocks[i][j].compute_fg_speeds(cx[i * nblocks + j], cy[i * nblocks + j]);
              blocks[i][j].limited_derivs();
            }
          }
        }

        #pragma omp taskwait

        // Compute step
        for (int i = 0; i < nblocks; i++) {
          for (int j = 0; j < nblocks; j++) {
            #pragma omp task
            blocks[i][j].compute_step(1, dt);
          }
        }

        #pragma omp taskwait

        // Evaluate dt
        t += dt;
      }
    }
  }
}

#endif /* BLOCKEDSIMULATION_H*/
