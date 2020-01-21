
#include <stdint.h>
#include "vednnLinearBackwardWeight.h"

#ifdef VEDNN_USE_OPENMP
#include <omp.h>
extern int __vednn_omp_num_threads ;
#endif


static inline vednnError_t
vednnLinearBackwardWeight_wrapper(
    vednnLinearBackwardWeight_t		pFunc,
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataIn,
    const void * 			pDataGradOut,
    void * 				pDataGradWeight
) {
    vednnError_t rc = VEDNN_SUCCESS ;
	#pragma omp parallel reduction(|:rc)
	{
		int64_t nthreads = omp_get_num_threads() ;
		int64_t threadid = omp_get_thread_num() ;

		int64_t nInDim = inDim / nthreads ;
		int64_t remain = inDim % nthreads ;

		int64_t inDimBegin = nInDim * threadid + ( threadid < remain ? threadid : remain ) ;
		int64_t myInDim    = nInDim + ( threadid < remain ? 1 : 0 ) ;

		if( nInDim == 0 ) {
			rc |= VEDNN_SUCCESS;
		} else {
			rc |= pFunc(inDim, outDim, nBatch, pDataIn, pDataGradOut, pDataGradWeight,
				inDimBegin, inDimBegin+myInDim) ;
		}
	}
    return rc;
}

/* ----------------------------------------------------------------------- */
vednnError_t vednnLinearBackwardWeight(
    const uint64_t			inDim,
    const uint64_t			outDim,
    const uint64_t			nBatch,
    const void * 			pDataIn,
    const void * 			pDataGradOut,
    void * 				pDataGradWeight
)
{
  // [todo] add variations
  {
    return vednnLinearBackwardWeight_wrapper(
	vednnLinearBackwardWeight_default,
	inDim, outDim, nBatch,
	pDataIn, pDataGradOut, pDataGradWeight ) ;
  }

}




