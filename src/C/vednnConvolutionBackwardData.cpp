
#define restrict
extern "C" {
#include "vednnConvolutionBackwardData.h"
}
#include <stdint.h>
#include "vednnLaunch.h"

//------------------------------------------------------------------------------
template<typename T>
struct vednnConvBwdFunctor {
	const vednnConvBackwardData_t			m_func;
	const vednnTensorParam_t* const			m_pO;
	const void* const						m_O;
	const vednnFilterParam_t* const			m_pW;
	const void*	const 						m_W;
	const vednnConvolutionParam_t* const	m_conv;
	const vednnTensorParam_t* const			m_pI;
	void*	const							m_I;

	inline vednnConvBwdFunctor(vednnConvBackwardData_t func, const vednnTensorParam_t* pO, const void* O,
		const vednnFilterParam_t* pW, const void* W, const vednnConvolutionParam_t* conv,
		const vednnTensorParam_t* pI, void* I) :
		m_func(func), m_pO(pO), m_O(O), m_pW(pW), m_W(W), m_conv(conv), m_pI(pI), m_I(I)
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

		return m_func(&pO, O, &pW, W, m_conv, &pI, I);
	}
};

//------------------------------------------------------------------------------
#define CALL(F) return vednn_launch_2d(pParamGradOut->batch, pParamGradOut->channel / pParamConv->group,\
	vednnConvBwdFunctor<float>(&F, pParamGradOut, pDataGradOut, pParamKernel, pDataKernel,\
		pParamConv, pParamGradIn, pDataGradIn))

//------------------------------------------------------------------------------
vednnError_t vednnConvolutionBackwardData(
    const vednnTensorParam_t*		pParamGradOut,
    const void*						pDataGradOut,
    const vednnFilterParam_t*		pParamKernel,
    const void*						pDataKernel,
    const vednnTensorParam_t*		pParamGradIn,
    void*							pDataGradIn,
    const vednnConvolutionParam_t*	pParamConv,
    vednnConvolutionAlgorithm_t		algo
) {
	if(algo == VEDNN_CONV_ALGORITHM_DIRECT) {
    	// [todo] add variations
		if(pParamGradIn->height * pParamGradIn->width <= 16 ||
			(pParamGradIn->height * pParamGradIn->width < 64
	  		&& pParamGradIn->height * pParamGradIn->width < pParamGradIn->channel)) {
	  		CALL(vednnConvolutionBackwardData_direct_vecC);
    	} else if(pParamConv->strideHeight == 1 && pParamConv->strideWidth == 1
			&& pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1) {
      		if(pParamGradIn->height == pParamGradOut->height && pParamGradIn->width == pParamGradOut->width) {
				if(pParamKernel->height == 5 && pParamKernel->width == 5) {
	  				CALL(vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker5);
				} else if(pParamKernel->height == 3 && pParamKernel->width == 3) {
	  				CALL(vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker3);
				} else if(pParamKernel->height == 2 && pParamKernel->width == 2) {
	  				CALL(vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker2);
				} else if(pParamKernel->height == 1 && pParamKernel->width == 1) {
	  				CALL(vednnConvolutionBackwardData_direct_dil1_str1_padsame_ker1);
				} else {
					CALL(vednnConvolutionBackwardData_direct_dil1_str1_padsame);
				}
      		} else if(pParamConv->padHeight == 0 && pParamConv->padWidth == 0
	    		&& pParamKernel->height == 3 && pParamKernel->width == 3
	    		&& (pParamGradIn->width & 0x01) == 0 && pParamGradIn->width <=256
	    		&& (pParamGradOut->width & 0x01) == 0
	    		&& (((uint64_t)pDataGradIn) & 0x07) == 0
	    		&& (((uint64_t)pDataGradOut) & 0x07) == 0) {
				if(pParamGradIn->width <= 32) {
	  				CALL(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU32_ow2X_ioaligned);
				} else {
	  				CALL(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iw2XU256_ow2X_ioaligned);
				}
      		} else if(pParamGradIn->width <= 128) {
				if(pParamConv->padHeight == 0 && pParamConv->padWidth == 0
	    			&& pParamKernel->height == 3 && pParamKernel->width == 3) {
	  				CALL(vednnConvolutionBackwardData_direct_dil1_str1_pad0_ker3_iwU128);
				} else {
					CALL(vednnConvolutionBackwardData_direct_dil1_str1_iwU128);
				}
      		} else {
				CALL(vednnConvolutionBackwardData_direct_dil1_str1);
      		}
		} else {
			if(pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
				&& pParamConv->padHeight == 0 && pParamConv->padWidth == 0
				&& pParamKernel->height == 1 && pParamKernel->width == 1
				&& pParamGradOut->width <= 128) {
				CALL(vednnConvolutionBackwardData_direct_dil1_pad0_ker1_owU128);
			} else if(pParamKernel->height == 5 && pParamKernel->width == 5
				&& pParamConv->dilationHeight == 1 && pParamConv->dilationWidth == 1
				&& pParamConv->strideHeight == 2 && pParamConv->strideWidth == 2
				&& pParamConv->padHeight == 2 && pParamConv->padWidth == 2) {
				if(pParamGradIn->width <= 128) {
					CALL(vednnConvolutionBackwardData_direct_dil1_str2_pad2_ker5_iwU128);
				} else {
					CALL(vednnConvolutionBackwardData_direct_dil1_str2_pad2_ker5);
				}
			} else if(pParamGradIn->width <= 128) {
				if(pParamKernel->height == 3 && pParamKernel->width == 3 ) {
					CALL(vednnConvolutionBackwardData_direct_ker3_iwU128);
				} else if( pParamKernel->height == 5 && pParamKernel->width == 5) {
					CALL(vednnConvolutionBackwardData_direct_ker5_iwU128);
				} else {
					CALL(vednnConvolutionBackwardData_direct_iwU128);
				}
			} else {
				if(pParamKernel->height == 5 && pParamKernel->width == 5) {
					CALL(vednnConvolutionBackwardData_direct_ker5);
				} else {
					CALL(vednnConvolutionBackwardData_direct_default);
				}
			}
		}
	} else {
		return VEDNN_ERROR_INVALID_PARAM ;
	}
}

