
#define restrict
#include <stdint.h>
#include "vednnLaunch.h"
extern "C" {
#include "vednnLinearForward.h"
}

//------------------------------------------------------------------------------
template<typename T>
struct vednnBGEMMFwdFunctor {
	const uint64_t	m_ic;
    const uint64_t	m_oc;
    const uint64_t	m_b;
    const T* const	m_I;
    const T* const	m_W;
    T* const		m_O;

	inline vednnBGEMMFwdFunctor(const uint64_t ic, const uint64_t oc, const uint64_t b,
		const void* I, const void* W, void* O) :
		m_ic(ic), m_oc(oc), m_b(b), m_I((const T*)I), m_W((const T*)W), m_O((T*)O)
	{}

	inline vednnLinearForward_t func(const uint64_t ic, const uint64_t oc, const uint64_t I, const uint64_t W, const uint64_t O) const {
		if(oc <= 32) {
	  		return &vednnLinearForward_oU32;
		} else {
    		if((oc & 0x01) == 0 && (W & 0x07) == 0 && (O & 0x07) == 0) {
	  			return &vednnLinearForward_o2X_woaligned;
    		} else {
	 			return &vednnLinearForward_default;
			}
  		}
	}

	vednnError_t operator()(const int min_c, const int max_c) const {
		auto I = m_I + m_ic * min_c * m_b;
		auto O = m_O + m_oc * min_c * m_b;
		auto W = m_W + m_ic * min_c * m_oc;
		auto f = func(m_ic, m_oc, (uint64_t)I, (uint64_t)W, (uint64_t)O);
		
		int rc = VEDNN_SUCCESS;
		for(int c = min_c; c < max_c; c++) {
			rc |= f(m_ic, m_oc, m_oc, m_b, I, W, O);
			I += m_ic * m_b;
			O += m_oc * m_b;
			W += m_ic * m_oc;
		}
		return (vednnError_t)rc;
	}
};

//------------------------------------------------------------------------------
vednnError_t vednnBGEMMForward(
    const uint64_t	inDim,
    const uint64_t	outDim,
    const uint64_t	nBatch,
    const uint64_t	cnt,
    const void*		pDataIn,
    const void*		pDataWeight,
    void*			pDataOut
) {
	return vednn_launch_1d(cnt, vednnBGEMMFwdFunctor<float>(inDim, outDim, nBatch, pDataIn, pDataWeight, pDataOut));
}

