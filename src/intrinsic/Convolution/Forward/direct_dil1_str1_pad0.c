#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionForward_direct_dil1_str1_pad0(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnFilterParam_t * restrict 	pParamKernel,
    const void * restrict 			pDataKernel,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnTensorParam_t * restrict 	pParamOut,
    void * restrict 				pDataOut
)
{
  const int64_t batch      = pParamIn->batch;
  const int64_t inChannel  = pParamIn->channel;
  const int64_t inWidth    = pParamIn->width;
  const int64_t inHeight   = pParamIn->height;
  const int64_t outChannel = pParamOut->channel;
  const int64_t outWidth   = pParamOut->width;
  const int64_t outHeight  = pParamOut->height;
  const int64_t kernWidth  = pParamKernel->width;
  const int64_t kernHeight = pParamKernel->height;

  const int64_t group          = pParamConv->group;
  //  const int64_t strideWidth    = pParamConv->strideWidth;		// must be 1
  //  const int64_t strideHeight   = pParamConv->strideHeight;		// must be 1
  //  const int64_t padWidth       = pParamConv->padWidth;		// must be 0
  //  const int64_t padHeight      = pParamConv->padHeight;		// must be 0
  //  const int64_t dilationWidth  = pParamConv->dilationWidth;		// must be 1
  //  const int64_t dilationHeight = pParamConv->dilationHeight;		// must be 1


  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * restrict pIn     = pDataIn;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pOut    = pDataOut;

  {
    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k) * outHeight*outWidth ;


	  for (int64_t y=0; y<outHeight; y++) {
	    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {
	      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

	      __vr vrsum = _vel_vbrds_vsl(0.0f, vl) ;
	      for (int64_t r = 0; r < kernHeight; r++) {
		for (int64_t s = 0; s < kernWidth; s++) {
		  const int64_t h = y + r ;
		  for (int64_t c = 0; c < inChannelGroup; c++) {
		    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		    __vr vrin = _vel_vldu_vssl(4,&pInChannel[h*inWidth+x0+s], vl) ;

		    const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		    vrsum = _vel_vfmads_vvsvl(vrsum, *pKerValue, vrin, vl) ;
		  } // inChannel
		} // kernWidth
	      } // kernHeight


	      _vel_vstu_vssl(vrsum, 4, pOut+outIndex, vl) ;

	      outIndex += vl ;
	    } // x
	  } // y

	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * outHeight*outWidth ;

	  for (int64_t y=0; y<outHeight; y++) {
	    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

	      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

	      __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	      for (int64_t r = 0; r < kernHeight; r++) {
		for (int64_t s = 0; s < kernWidth; s++) {

		  const int64_t h = y + r ;
		  for (int64_t c = 0; c < inChannelGroup; c++) {
		    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		    __vr vrin = _vel_vldu_vssl(4,&pInChannel[h*inWidth+x0+s], vl) ;
		    __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

		    const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
							       pKerValue+      inChannelGroup * kernHeight * kernWidth) ;

		    vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
		  } // inChannel
		} // kernWidth
	      } // kernHeight

	      _vel_vstu_vssl(vrsum01, 4, pOut+outIndex0, vl) ;
	      _vel_vstl_vssl(vrsum01, 4, pOut+outIndex1, vl) ;

	      outIndex0 += vl ;
	      outIndex1 += vl ;
	    } // x
	  } // y

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * outHeight*outWidth ;
	  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * outHeight*outWidth ;
	  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * outHeight*outWidth ;

	  for (int64_t y=0; y<outHeight; y++) {
	    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

	      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

	      __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	      __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;

	      for (int64_t r = 0; r < kernHeight; r++) {
		for (int64_t s = 0; s < kernWidth; s++) {
		  const int64_t h = y + r ;
		    for (int64_t c = 0; c < inChannelGroup; c++) {
		    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		    __vr vrin = _vel_vldu_vssl(4,&pInChannel[h*inWidth+x0+s], vl) ;
		    __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

		    const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
							       pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
		    const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
							       pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;

		    vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
		    vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;
		  } // inChannel
		} // kernWidth
	      } // kernHeight

	      _vel_vstu_vssl(vrsum01, 4, pOut+outIndex0, vl) ;
	      _vel_vstl_vssl(vrsum01, 4, pOut+outIndex1, vl) ;
	      _vel_vstu_vssl(vrsum23, 4, pOut+outIndex2, vl) ;
	      _vel_vstl_vssl(vrsum23, 4, pOut+outIndex3, vl) ;

	      outIndex0 += vl ;
	      outIndex1 += vl ;
	      outIndex2 += vl ;
	      outIndex3 += vl ;
	    } // x
	  } // y

	  k+=4 ;
	}
	for (; k < outChannelGroup; k+=8) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * outHeight*outWidth ;
	  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * outHeight*outWidth ;
	  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * outHeight*outWidth ;
	  int64_t outIndex4 = outGroupOffset + (n * outChannel + k+4) * outHeight*outWidth ;
	  int64_t outIndex5 = outGroupOffset + (n * outChannel + k+5) * outHeight*outWidth ;
	  int64_t outIndex6 = outGroupOffset + (n * outChannel + k+6) * outHeight*outWidth ;
	  int64_t outIndex7 = outGroupOffset + (n * outChannel + k+7) * outHeight*outWidth ;

	  for (int64_t y=0; y<outHeight; y++) {
	    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

	      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

	      __vr vrsum01 = _vel_pvbrd_vsl(0UL, vl) ;
	      __vr vrsum23 = _vel_pvbrd_vsl(0UL, vl) ;
	      __vr vrsum45 = _vel_pvbrd_vsl(0UL, vl) ;
	      __vr vrsum67 = _vel_pvbrd_vsl(0UL, vl) ;

	      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth ;

	      for (int64_t r = 0; r < kernHeight; r++) {
		for (int64_t s = 0; s < kernWidth; s++) {

		  const int64_t h = y + r ;
		  for (int64_t c = 0; c < inChannelGroup; c++) {
		    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;


		    __vr vrin = _vel_vldu_vssl(4,&pInChannel[h*inWidth+x0+s], vl) ;
		    __vr vrinP = _vel_vshf_vvvsl(vrin, vrin, VE_VSHUFFLE_YUZU, vl) ;

		    const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		    const uint64_t kerValue01 = _vel_pack_f32p(pKerValue,
							       pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
		    const uint64_t kerValue23 = _vel_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
							       pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;
		    const uint64_t kerValue45 = _vel_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth,
							       pKerValue + 5 * inChannelGroup * kernHeight * kernWidth) ;
		    const uint64_t kerValue67 = _vel_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth,
							       pKerValue + 7 * inChannelGroup * kernHeight * kernWidth) ;

		    vrsum01 = _vel_pvfmad_vvsvl(vrsum01, kerValue01, vrinP, vl) ;
		    vrsum23 = _vel_pvfmad_vvsvl(vrsum23, kerValue23, vrinP, vl) ;
		    vrsum45 = _vel_pvfmad_vvsvl(vrsum45, kerValue45, vrinP, vl) ;
		    vrsum67 = _vel_pvfmad_vvsvl(vrsum67, kerValue67, vrinP, vl) ;
		  } // inChannel
		} // kernWidth
	      } // kernHeight

	      _vel_vstu_vssl(vrsum01, 4, pOut+outIndex0, vl) ;
	      _vel_vstl_vssl(vrsum01, 4, pOut+outIndex1, vl) ;
	      _vel_vstu_vssl(vrsum23, 4, pOut+outIndex2, vl) ;
	      _vel_vstl_vssl(vrsum23, 4, pOut+outIndex3, vl) ;
	      _vel_vstu_vssl(vrsum45, 4, pOut+outIndex4, vl) ;
	      _vel_vstl_vssl(vrsum45, 4, pOut+outIndex5, vl) ;
	      _vel_vstu_vssl(vrsum67, 4, pOut+outIndex6, vl) ;
	      _vel_vstl_vssl(vrsum67, 4, pOut+outIndex7, vl) ;

	      outIndex0 += vl ;
	      outIndex1 += vl ;
	      outIndex2 += vl ;
	      outIndex3 += vl ;
	      outIndex4 += vl ;
	      outIndex5 += vl ;
	      outIndex6 += vl ;
	      outIndex7 += vl ;
	    } // x
	  } // y
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
