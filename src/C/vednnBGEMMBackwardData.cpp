
#define restrict
#include <stdint.h>
#include "vednnLaunch.h"
extern "C" {
#include "vednnLinearBackwardData.h"
}

//------------------------------------------------------------------------------
template<typename T>
struct vednnBGEMMBwdFunctor {
	const uint64_t	m_ic;
    const uint64_t	m_oc;
    const uint64_t	m_b;
    const T* const	m_O;
    const T* const	m_W;
    T* const		m_I;

	inline vednnBGEMMBwdFunctor(const uint64_t ic, const uint64_t oc, const uint64_t b,
		const void* O, const void* W, void* I) :
		m_ic(ic), m_oc(oc), m_b(b), m_O((const T*)O), m_W((const T*)W), m_I((T*)I)
	{}

	inline vednnLinearBackwardData_t func(const uint64_t ic, const uint64_t oc, const uint64_t I, const uint64_t W, const uint64_t O) const {
		if(oc <= 128 && ic >= 256) {
			if((oc&0x1)==0 && (W & 0x7)==0) {
				return &vednnLinearBackwardData_o2XU128_waligned;
			} else {
				return &vednnLinearBackwardData_oU128;
			}
		} else if(oc <= 256) {
			return &vednnLinearBackwardData_oU256;
		} else if(((oc & 0x1) == 0) && ((W&0x7)==0) && ((O&0x7)==0)) {
			return &vednnLinearBackwardData_o2X_woaligned;
		} else {
			return &vednnLinearBackwardData_default;
		}		
	}

	vednnError_t operator()(const int min_c, const int max_c) const {
		auto I = m_I + m_ic * m_b * min_c;
		auto O = m_O + m_oc * m_b * min_c;
		auto W = m_W + m_ic * m_oc * min_c;

		auto f = func(m_ic, m_oc, (uint64_t)I, (uint64_t)W, (uint64_t)O);
		int rc = VEDNN_SUCCESS;
		for(int c = min_c; c < max_c; c++) {
			rc |= f(m_ic, m_oc, m_ic, m_b, O, W, I);
			I += m_ic * m_b;
			O += m_oc * m_b;
			W += m_ic * m_oc;
		}
		return (vednnError_t)rc;
	}
};

//------------------------------------------------------------------------------
vednnError_t vednnBGEMMBackwardData(
    const uint64_t	inDim,
    const uint64_t	outDim,
    const uint64_t	nBatch,
    const uint64_t	cnt,
    const void*		pDataGradOut,
    const void*		pDataWeight,
    void*			pDataGradIn
) {
	//return vednn_launch_2d(nBatch, outDim, vednnLinearBwdFunctor<float>(inDim, outDim, nBatch, pDataGradOut, pDataWeight, pDataGradIn));
	return vednn_launch_1d(cnt, vednnBGEMMBwdFunctor<float>(inDim, outDim, nBatch, pDataGradOut, pDataWeight, pDataGradIn));
}

