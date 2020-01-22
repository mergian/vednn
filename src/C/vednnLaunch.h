#pragma once

#include "vednn.h"

#ifdef VEDNN_USE_OPENMP
#include <omp.h>
#endif

#include <algorithm>

/** TODO: Also 2d parallelize Linear. Be careful, there are calls that require
 * the OC to be a even number! */

//------------------------------------------------------------------------------
template<typename F>
inline vednnError_t vednn_launch_1d(const int cnt, F func) {
	int rc = VEDNN_SUCCESS;
	if(cnt == 1) {
		return func(0, cnt);
	} else {
		#pragma omp parallel reduction(|:rc)
		{
			int nthreads = omp_get_num_threads();
			int tcnt = (cnt + nthreads - 1) / nthreads;
			
			int tx = omp_get_thread_num();
			int min = tx * tcnt;
			int max = std::min((tx+1) * tcnt, cnt);

			if(min < max)
				rc |= (int)func(min, max);
		}
	}

	return (vednnError_t)rc;
}

//------------------------------------------------------------------------------
template<typename F>
inline vednnError_t vednn_launch_2d(const int x, const int y, F func) {
	int rc = VEDNN_SUCCESS;
	int cnt = x * y;
	if(cnt == 1) {
		return func(0, x, 0, y);
	} else {
		#pragma omp parallel reduction(|:rc)
		{
			int nthreads = omp_get_num_threads();
			int tx = omp_get_thread_num();

			if(x > (nthreads/2)) {
				int xcnt = (x + nthreads - 1) / nthreads;

				int min_x = tx * xcnt;
				int max_x = std::min((tx+1) * xcnt, x);
			
				if(min_x < max_x)
					rc |= func(min_x, max_x, 0, y);
			} else {
				int xthreads = x;
				int ythreads = nthreads / x;
				int ty = tx / xthreads;
				tx = tx % xthreads;

				int min_x = tx % xthreads;
				int max_x = min_x + 1;

				int ycnt = (y + ythreads - 1) / ythreads;
				int min_y = ty * ycnt;
				int max_y = std::min((ty+1) * ycnt, y);

				if(min_y < max_y)
					rc |= func(min_x, max_x, min_y, max_y);
			}
		}
	}

	return (vednnError_t)rc;
}

//------------------------------------------------------------------------------
