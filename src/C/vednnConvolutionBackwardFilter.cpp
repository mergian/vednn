#define restrict
extern "C" {
#include "vednnConvolutionBackwardFilter.h"
}
#include <stdint.h>
#include "vednnLaunch.h"

//------------------------------------------------------------------------------
template<typename T>
struct vednnConvFilFunctor {
	const vednnConvBackwardFilter_t			m_func;
	const vednnTensorParam_t* const			m_pI;
	const void*	const						m_I;
	const vednnTensorParam_t* const			m_pO;
	const void* const						m_O;
	const vednnConvolutionParam_t* const	m_conv;
	const vednnFilterParam_t* const			m_pW;
	void* const 							m_W;

	inline vednnConvFilFunctor(vednnConvBackwardFilter_t func, const vednnTensorParam_t* pI, const void* I,
		const vednnTensorParam_t* pO, const void* O, const vednnConvolutionParam_t* conv,
		const vednnFilterParam_t* pW, void* W) :
		m_func(func), m_pI(pI), m_I(I), m_pO(pO), m_O(O), m_conv(conv), m_pW(pW), m_W(W)
	{}

	vednnError_t operator()(const int min_oc, const int max_oc) const {
		return m_func(m_pI, m_I, m_pO, m_O, m_conv, m_pW, m_W, min_oc, max_oc - min_oc);
	}
};

//------------------------------------------------------------------------------
#define CALL(F) return vednn_launch_1d(pParamGradKernel->outChannel / pParamConv->group,\
	vednnConvFilFunctor<float>(&F, pParamIn, pDataIn, pParamGradOut, pDataGradOut, \
		pParamConv, pParamGradKernel, pDataGradKernel));

//------------------------------------------------------------------------------
vednnError_t vednnConvolutionBackwardFilter(
	const vednnTensorParam_t*		pParamIn,
	const void*						pDataIn,
	const vednnTensorParam_t*		pParamGradOut,
	const void*						pDataGradOut,
	const vednnFilterParam_t*		pParamGradKernel,
	void*							pDataGradKernel,
	const vednnConvolutionParam_t*	pParamConv,
	vednnConvolutionAlgorithm_t		algo
) {
	if(algo == VEDNN_CONV_ALGORITHM_DIRECT) {
		// [todo] add variations
		if(pParamGradOut->height * pParamGradOut->width <= 16 ||
			(pParamGradOut->height * pParamGradOut->width < 64
	  		&& pParamGradOut->height * pParamGradOut->width < pParamIn->channel)) {
			CALL(vednnConvolutionBackwardFilter_direct_vecC);
		} else if(pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1
			&& pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
			&& pParamIn->height == pParamGradOut->height
			&& pParamIn->width == pParamGradOut->width) {
			if(pParamGradKernel->height == 1 && pParamGradKernel->width == 1) {
				CALL(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker1);
			} else if(pParamGradKernel->height == 3 && pParamGradKernel->width == 3) {
				if(pParamGradOut->width * pParamGradOut->height <= 256) {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3_ohwU256);
				} else if(pParamGradOut->width <= 128) {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3_owU128);
				} else {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3);
				}
			} else if(pParamGradKernel->height == 5 && pParamGradKernel->width == 5) {
				if(pParamGradOut->width <= 128) {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker5_owU128);
				} else {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker5);
				}
			} else if(pParamGradKernel->height == 2 && pParamGradKernel->width == 2) {
				if(pParamGradOut->width <= 128) {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker2_owU128);
				} else {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker2);
				}
			} else {
				CALL(vednnConvolutionBackwardFilter_direct_dil1_str1_padsame);
			}
		} else if (pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
			&& pParamConv->padHeight == 0 && pParamConv->padWidth == 0
			&& pParamGradOut->height == (pParamIn->height - pParamGradKernel->height) / pParamConv->strideHeight + 1
			&& pParamGradOut->width == (pParamIn->width - pParamGradKernel->width) / pParamConv->strideWidth + 1) {
			if(pParamGradKernel->height == 3 && pParamGradKernel->width == 3
				&& pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1
				&& pParamIn->width <= 256
				&& (pParamIn->width & 0x01) == 0 && (((uint64_t)pDataIn) & 0x07) == 0
				&& (pParamGradOut->width & 0x01) == 0 && (((uint64_t)pDataGradOut) & 0x07) == 0) {
				CALL(vednnConvolutionBackwardFilter_direct_dil1_str1_pad0_ker3_ow2X_iw2XU256_igoaligned);
			} else if(pParamGradOut->width <= 128 && pParamGradKernel->height == 3 && pParamGradKernel->width == 3) {
				if(pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1) {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_str1_pad0_ker3_owU128);
				} else {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker3_owU128);
				}
			} else if(pParamGradKernel->height == 1 && pParamGradKernel->width == 1) {
				if(pParamGradOut->height * pParamGradOut->width <= 64) {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU64);
				} else if(pParamGradOut->height * pParamGradOut->width <= 128) {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_ohwU128);
				} else if(pParamGradOut->width <= 32) {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1_owU32);
				} else {
					CALL(vednnConvolutionBackwardFilter_direct_dil1_pad0_ker1);
				}
			} else if(pParamGradOut->width <= 32) {
				CALL(vednnConvolutionBackwardFilter_direct_dil1_pad0_owU32);
			} else {
				CALL(vednnConvolutionBackwardFilter_direct_dil1_pad0);
			}
		} else {
			if(pParamGradOut->width <= 128) {
				CALL(vednnConvolutionBackwardFilter_direct_owU128);
			} else {
				CALL(vednnConvolutionBackwardFilter_direct_default);
			}
		}
	} else {
		return VEDNN_ERROR_INVALID_PARAM;
	}
}

