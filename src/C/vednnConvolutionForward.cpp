#define restrict
extern "C" {
#include "vednnConvolutionForward.h"
}
#include <stdint.h>
#include "vednnLaunch.h"

//------------------------------------------------------------------------------
template<typename T>
struct vednnConvFwdFunctor {
	const vednnConvForward_t				m_func;
	const vednnTensorParam_t* const			m_pI;
	const void*	const						m_I;
	const vednnFilterParam_t* const			m_pW;
	const void*	const 						m_W;
	const vednnConvolutionParam_t* const	m_conv;
	const vednnTensorParam_t* const			m_pO;
	void* const								m_O;

	inline vednnConvFwdFunctor(vednnConvForward_t func, const vednnTensorParam_t* pI, const void* I,
		const vednnFilterParam_t* pW, const void* W, const vednnConvolutionParam_t* conv,
		const vednnTensorParam_t* pO, void* O) :
		m_func(func), m_pI(pI), m_I(I), m_pW(pW), m_W(W), m_conv(conv), m_pO(pO), m_O(O)
	{}

	vednnError_t operator()(const int min_b, const int max_b, const int min_oc, const int max_oc) const {
		vednnTensorParam_t pI  = *m_pI;
		pI.batch = max_b - min_b;

		vednnTensorParam_t pO = *m_pO;
		pO.batch = max_b - min_b;
		// don't set the channel here, so we know the stride!

		vednnFilterParam_t pW = *m_pW;
		pW.outChannel	= max_oc - min_oc;

		auto* I = ((T*)m_I) + min_b  * m_pI->channel * m_pI->height * m_pI->width;
		auto* O = ((T*)m_O) + (min_b * m_pO->channel + min_oc) * m_pO->height * m_pO->width;
		auto* W = ((T*)m_W) + min_oc * m_pW->inChannel * m_pW->height * m_pW->width;

		return m_func(&pI, I, &pW, W, m_conv, &pO, O);
	}
};

//------------------------------------------------------------------------------
#define CALL(F) return vednn_launch_2d(pParamOut->batch, pParamOut->channel / pParamConv->group,\
	vednnConvFwdFunctor<float>(&F, pParamIn, pDataIn, pParamKernel, pDataKernel,\
		pParamConv, pParamOut, pDataOut))

//------------------------------------------------------------------------------
vednnError_t vednnConvolutionForward(
	const vednnTensorParam_t*		pParamIn,
	const void*						pDataIn,
	const vednnFilterParam_t*		pParamKernel,
	const void*						pDataKernel,
	const vednnTensorParam_t*		pParamOut,
	void*							pDataOut,
	const vednnConvolutionParam_t*	pParamConv,
	vednnConvolutionAlgorithm_t		algo
) {
	if(algo == VEDNN_CONV_ALGORITHM_DIRECT) {
		// [todo] add variations
		if(pParamOut->height * pParamOut->width <= 16) {
			CALL(vednnConvolutionForward_direct_vecC);
		} else if(pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1
					&& pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
					&& pParamIn->height == pParamOut->height
					&& pParamIn->width == pParamOut->width) {
			if(pParamKernel->width == 1 && pParamKernel->height == 1)  {
				CALL(vednnConvolutionForward_direct_dil1_str1_pad0_ker1);
			} else if(pParamKernel->height == 3 && pParamKernel->width == 3) {
				if(pParamIn->channel == pParamConv->group) {// aka inputChannelGroup==1
					if(pParamOut->width <= 128) {
						CALL(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1_owU128);
					} else {
						CALL(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1);
					}
				} else if(pParamKernel->inChannel % 1024 == 0) {
					CALL(vednnConvolutionForward_direct_dil1_str1_padsame_ker3_c1024x);
				} else {
					CALL(vednnConvolutionForward_direct_dil1_str1_padsame_ker3);
				}
			} else if(pParamKernel->height == 5 && pParamKernel->width == 5) {
				if(pParamOut->width <= 128 ) {
					CALL(vednnConvolutionForward_direct_dil1_str1_padsame_ker5_owU128);
				} else {
					CALL(vednnConvolutionForward_direct_dil1_str1_padsame_ker5);
				}
			} else if(pParamKernel->height == 2 && pParamKernel->width == 2) {
				CALL(vednnConvolutionForward_direct_dil1_str1_padsame_ker2);
			} else {
				CALL(vednnConvolutionForward_direct_dil1_str1_padsame);
			}
		} else if(pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
			&& pParamConv->padHeight == 0  && pParamConv->padWidth == 0
			&& pParamOut->height == (pParamIn->height - pParamKernel->height) / pParamConv->strideHeight + 1
			&& pParamOut->width == (pParamIn->width - pParamKernel->width) / pParamConv->strideWidth + 1) {
			if(pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1) {
				if(pParamKernel->height == 3 && pParamKernel->width == 3
					&& (pParamIn->width <= 256)
					&& (pParamIn->width & 0x1) == 0  && (((uint64_t)pDataIn) & 0x7) == 0
					&& (pParamOut->width & 0x1) == 0 && (((uint64_t)pDataOut) & 0x7) == 0) {
					CALL(vednnConvolutionForward_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned);
				} else if (pParamOut->width <= 128) {
					CALL(vednnConvolutionForward_direct_dil1_str1_pad0_owU128);
				} else {
					CALL(vednnConvolutionForward_direct_dil1_str1_pad0);
				}
			} else {
				if(pParamKernel->width == 1 && pParamKernel->height == 1) {
					if(pParamOut->width <= 128) {
						CALL(vednnConvolutionForward_direct_dil1_pad0_owU128_ker1);
					} else {
						CALL(vednnConvolutionForward_direct_dil1_pad0_ker1);
					}
				} else {
					if(pParamOut->width <= 128) {
						CALL(vednnConvolutionForward_direct_dil1_pad0_owU128);
					} else {
						CALL(vednnConvolutionForward_direct_dil1_pad0);
					}
				}
			}
		} else {
			if(pParamOut->width <= 128) {
				CALL(vednnConvolutionForward_direct_owU128);
			} else {
				CALL(vednnConvolutionForward_direct_default);
			}
		}
  	} else {
		return VEDNN_ERROR_INVALID_PARAM ;
	}
}

