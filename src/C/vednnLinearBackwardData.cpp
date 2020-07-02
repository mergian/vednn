
#define restrict
#include <stdint.h>
#include "vednnLaunch.h"
extern "C" {
#include "vednnLinearBackwardData.h"
}

//------------------------------------------------------------------------------
template<typename T>
struct vednnLinearBwdFunctor {
	const uint64_t		m_ic;
    const uint64_t		m_oc;
    const uint64_t		m_b;
    const uint64_t		m_bgemm;
    const void* const	m_O;
    const void* const	m_W;
    void* const			m_I;

	inline vednnLinearBwdFunctor(const uint64_t ic, const uint64_t oc, const uint64_t b, const uint64_t bgemm,
		const void* O, const void* W, void* I) :
		m_ic(ic), m_oc(oc), m_b(b), m_bgemm(bgemm), m_O(O), m_W(W), m_I(I)
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

	vednnError_t operator()(const int min, const int max) const {
		if(m_bgemm == 1) {
			auto min_ic = min;
			auto max_ic = max;
			auto cnt_ic = max_ic - min_ic;
			auto I = ((T*)m_I) + min_ic;
			auto O = ((T*)m_O);
			auto W = ((T*)m_W) + min_ic * m_oc;
			auto f = func(cnt_ic, m_oc, (uint64_t)I, (uint64_t)W, (uint64_t)O);
			return f(m_ic, m_oc, cnt_ic, m_b, O, W, I);
		} else {
			auto I = (T*)m_I;
			auto O = (T*)m_O;
			auto W = (T*)m_W;
			auto f = func(m_ic, m_oc, (uint64_t)I, (uint64_t)W, (uint64_t)O);

			int rc = 0;
			for(int i = min; i < max; i++) {
				rc |= f(m_ic, m_oc, m_ic, m_b, O, W, I);
				I += m_ic * m_b;
				O += m_oc * m_b;
				W += m_oc * m_ic;
			}

			return (vednnError_t)rc;
		}
	}
};

//------------------------------------------------------------------------------
vednnError_t vednnLinearBackwardData(
    const uint64_t	inDim,
    const uint64_t	outDim,
    const uint64_t	nBatch,
    const uint64_t	bgemm,
    const void*		pDataGradOut,
    const void*		pDataWeight,
    void*			pDataGradIn
) {
	return vednn_launch_1d(bgemm == 1 ? inDim : bgemm, vednnLinearBwdFunctor<float>(inDim, outDim, nBatch, bgemm, pDataGradOut, pDataWeight, pDataGradIn));
}
