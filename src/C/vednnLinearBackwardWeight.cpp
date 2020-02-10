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
    const uint64_t		m_bgemm;
    const void* const	m_I;
    const void* const	m_O;
    void* const			m_W;

	inline vednnLinearFilFunctor(const uint64_t ic, const uint64_t oc, const uint64_t b, const uint64_t bgemm,
		const void* I, const void* O, void* W) :
		m_ic(ic), m_oc(oc), m_b(b), m_bgemm(bgemm), m_I(I), m_O(O), m_W(W)
	{}

	vednnError_t operator()(const int min_ic, const int max_ic) const {
		if(m_bgemm == 1)
			return vednnLinearBackwardWeight_default(m_ic, m_oc, m_b, m_I, m_O, m_W, min_ic, max_ic);

		auto I = (T*)m_I;
		auto O = (T*)m_O;
		auto W = (T*)m_W;

		int rc = 0;
		for(int i = min_ic; i < max_ic; i++) {
			rc |= vednnLinearBackwardWeight_default(m_ic, m_oc, m_b, I, O, W, 0, m_ic);
			I += m_ic * m_b;
			O += m_oc * m_b;
			W += m_oc * m_ic;
		}

		return (vednnError_t)rc;
	}
};

//------------------------------------------------------------------------------
vednnError_t vednnLinearBackwardWeight(
    const uint64_t	inDim,
    const uint64_t	outDim,
    const uint64_t	nBatch,
    const uint64_t	bgemm,
    const void* 	pDataIn,
    const void* 	pDataGradOut,
    void* 			pDataGradWeight
) {
	return vednn_launch_1d(bgemm == 1 ? inDim : bgemm, vednnLinearFilFunctor<float>(inDim, outDim, nBatch, bgemm, pDataIn, pDataGradOut, pDataGradWeight));
}




