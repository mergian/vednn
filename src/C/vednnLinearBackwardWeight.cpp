#define restrict
#include <stdint.h>
#include "vednnLaunch.h"
extern "C" {
#include "vednnLinearBackwardWeight.h"
}

//------------------------------------------------------------------------------
template<typename T>
struct vednnLinearFilFunctor {
	const uint64_t		m_ic;
    const uint64_t		m_oc;
    const uint64_t		m_b;
    const void* const	m_I;
    const void* const	m_O;
    void* const			m_W;

	inline vednnLinearFilFunctor(const uint64_t ic, const uint64_t oc, const uint64_t b,
		const void* I, const void* O, void* W) :
		m_ic(ic), m_oc(oc), m_b(b), m_I(I), m_O(O), m_W(W)
	{}

	vednnError_t operator()(const int min_ic, const int max_ic) const {
		return vednnLinearBackwardWeight_default(m_ic, m_oc, m_b, m_I, m_O, m_W, min_ic, max_ic);
	}
};

//------------------------------------------------------------------------------
vednnError_t vednnLinearBackwardWeight(
    const uint64_t	inDim,
    const uint64_t	outDim,
    const uint64_t	nBatch,
    const void* 	pDataIn,
    const void* 	pDataGradOut,
    void* 			pDataGradWeight,
    const int		parallel
) {
	vednnLinearFilFunctor<float> op(inDim, outDim, nBatch, pDataIn, pDataGradOut, pDataGradWeight);
	return parallel ? vednn_launch_1d(inDim, op) : op(0, inDim);
}




