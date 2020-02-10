
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
    const void* const	m_O;
    const void* const	m_W;
    void* const			m_I;

	inline vednnLinearBwdFunctor(const uint64_t ic, const uint64_t oc, const uint64_t b,
		const void* O, const void* W, void* I) :
		m_ic(ic), m_oc(oc), m_b(b), m_O(O), m_W(W), m_I(I)
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

	vednnError_t operator()(const int min_b, const int max_b, const int min_oc, const int max_oc) const {
		int b = max_b - min_b;
		int oc = max_oc - min_oc;
		auto I = ((T*)m_I) + m_ic * min_b;
		auto O = ((T*)m_O) + m_oc * min_b + min_oc;
		auto W = ((T*)m_W) + min_oc;
		auto f = func(m_ic, oc, (uint64_t)I, (uint64_t)W, (uint64_t)O);
		return f(m_ic, m_oc, oc, b, O, W, I);
	}

	vednnError_t operator()(const int min_ic, const int max_ic) const {
		int ic = max_ic - min_ic;
		auto I = ((T*)m_I) + min_ic;
		auto O = ((T*)m_O);
		auto W = ((T*)m_W) + min_ic * m_oc;
		auto f = func(ic, m_oc, (uint64_t)I, (uint64_t)W, (uint64_t)O);
		return f(m_ic, m_oc, ic, m_b, O, W, I);
	}
};

//------------------------------------------------------------------------------
vednnError_t vednnLinearBackwardData(
    const uint64_t	inDim,
    const uint64_t	outDim,
    const uint64_t	nBatch,
    const void*		pDataGradOut,
    const void*		pDataWeight,
    void*			pDataGradIn,
    const int		parallel
) {
	//return vednn_launch_2d(nBatch, outDim, vednnLinearBwdFunctor<float>(inDim, outDim, nBatch, pDataGradOut, pDataWeight, pDataGradIn));
	vednnLinearBwdFunctor<float> op(inDim, outDim, nBatch, pDataGradOut, pDataWeight, pDataGradIn);
	return parallel ? vednn_launch_1d(inDim, op) : op(0, inDim);
}

