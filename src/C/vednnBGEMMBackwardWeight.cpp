#define restrict
#include <stdint.h>
#include "vednnLaunch.h"
extern "C" {
#include "vednnLinearBackwardWeight.h"
}

//------------------------------------------------------------------------------
template<typename T>
struct vednnBGEMMFilFunctor {
	const uint64_t	m_ic;
    const uint64_t	m_oc;
    const uint64_t	m_b;
    const T* const	m_I;
    const T* const	m_O;
    T* const		m_W;

	inline vednnBGEMMFilFunctor(const uint64_t ic, const uint64_t oc, const uint64_t b,
		const void* I, const void* O, void* W) :
		m_ic(ic), m_oc(oc), m_b(b), m_I((const T*)I), m_O((const T*)O), m_W((T*)W)
	{}

	vednnError_t operator()(const int min_c, const int max_c) const {
        auto I = m_I + m_ic * min_c * m_b;
		auto O = m_O + m_oc * min_c * m_b;
		auto W = m_W + m_ic * min_c * m_oc;

        int rc = VEDNN_SUCCESS;
        for(int c = min_c; c < max_c; c++) {
		    rc |= vednnLinearBackwardWeight_default(m_ic, m_oc, m_b, m_I, m_O, m_W, 0, m_ic);
            I += m_ic * m_b;
            O += m_oc * m_b;
            W += m_ic * m_oc;
        }
        return (vednnError_t)rc;
	}
};

//------------------------------------------------------------------------------
vednnError_t vednnLinearBackwardWeight(
    const uint64_t	inDim,
    const uint64_t	outDim,
    const uint64_t	nBatch,
    const uint64_t	cnt,
    const void* 	pDataIn,
    const void* 	pDataGradOut,
    void* 			pDataGradWeight
) {
	return vednn_launch_1d(cnt, vednnBGEMMFilFunctor<float>(inDim, outDim, nBatch, pDataIn, pDataGradOut, pDataGradWeight));
}




