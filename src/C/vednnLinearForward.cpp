
#define restrict
#include <stdint.h>
#include "vednnLaunch.h"
extern "C" {
#include "vednnLinearForward.h"
}

//------------------------------------------------------------------------------
template<typename T>
struct vednnLinearFwdFunctor {
	const uint64_t		m_ic;
    const uint64_t		m_oc;
    const uint64_t		m_b;
    const uint64_t		m_bgemm;
    const void* const	m_I;
    const void* const	m_W;
    void* const			m_O;

	inline vednnLinearFwdFunctor(const uint64_t ic, const uint64_t oc, const uint64_t b, const uint64_t bgemm,
		const void* I, const void* W, void* O) :
		m_ic(ic), m_oc(oc), m_b(b), m_bgemm(bgemm), m_I(I), m_W(W), m_O(O)
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

	vednnError_t operator()(const int min_b, const int max_b, const int min_oc, const int max_oc) const {
		assert(m_bgemm == 1);
		int b = max_b - min_b;
		int oc = max_oc - min_oc;
		auto I = ((T*)m_I) + m_ic * min_b;
		auto O = ((T*)m_O) + m_oc * min_b + min_oc;
		auto W = ((T*)m_W) + min_oc;
		auto f = func(m_ic, oc, (uint64_t)I, (uint64_t)W, (uint64_t)O);
		return f(m_ic, m_oc, oc, b, I, W, O);
	}

	vednnError_t operator()(const int min_oc, const int max_oc) const {
		if(m_bgemm == 1) {
			int oc = max_oc - min_oc;
			auto I = ((T*)m_I);
			auto O = ((T*)m_O) + min_oc;
			auto W = ((T*)m_W) + min_oc;
			auto f = func(m_ic, oc, (uint64_t)I, (uint64_t)W, (uint64_t)O);
			return f(m_ic, m_oc, oc, m_b, I, W, O);
		}

		auto I = (T*)m_I;
		auto O = (T*)m_O;
		auto W = (T*)m_W;
		auto f = func(m_ic, m_oc, (uint64_t)I, (uint64_t)W, (uint64_t)O);

		int rc = 0;
		for(int i = min_oc; i < max_oc; i++) {
			rc |= f(m_ic, m_oc, m_oc, m_b, I, W, O);
			I += m_ic * m_b;
			O += m_oc * m_b;
			W += m_oc * m_ic;
		}

		return (vednnError_t)rc;
	}
};

//------------------------------------------------------------------------------
vednnError_t vednnLinearForward(
    const uint64_t	inDim,
    const uint64_t	outDim,
    const uint64_t	nBatch,
    const uint64_t	bgemm,
    const void*		pDataIn,
    const void*		pDataWeight,
    void*			pDataOut
) {
	//return vednn_launch_2d(nBatch, outDim, vednnLinearFwdFunctor<float>(inDim, outDim, nBatch, pDataIn, pDataWeight, pDataOut));
	return vednn_launch_1d(bgemm == 1 ? outDim : bgemm, vednnLinearFwdFunctor<float>(inDim, outDim, nBatch, bgemm, pDataIn, pDataWeight, pDataOut));
}

