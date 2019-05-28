#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardFilter_direct_owU128(
    const vednnTensorParam_t * restrict 	pParamIn,
    const void * restrict 			pDataIn,
    const vednnTensorParam_t * restrict 	pParamGradOut,
    const void * restrict 			pDataGradOut,
    const vednnConvolutionParam_t * restrict 	pParamConv,
    const vednnFilterParam_t * restrict 	pParamGradKernel,
    void * restrict 				pDataGradKernel
#ifdef VEDNN_USE_OPENMP
    ,
    const int64_t				beginOChannel,
    const int64_t				nOChannel
#endif
)
{
  const int64_t inChannel   = pParamIn->channel;
  const int64_t inWidth     = pParamIn->width;
  const int64_t inHeight    = pParamIn->height;
  const int64_t batch       = pParamGradOut->batch;
  const int64_t gOutChannel = pParamGradOut->channel;
  const int64_t gOutWidth   = pParamGradOut->width;
  const int64_t gOutHeight  = pParamGradOut->height;
  const int64_t gKernWidth  = pParamGradKernel->width;
  const int64_t gKernHeight = pParamGradKernel->height;

  const int64_t group          = pParamConv->group;
  const int64_t strideWidth    = pParamConv->strideWidth;;
  const int64_t strideHeight   = pParamConv->strideHeight;
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
  const int64_t dilationWidth  = pParamConv->dilationWidth;
  const int64_t dilationHeight = pParamConv->dilationHeight;

  const int64_t inChannelGroup   =  inChannel   / group;
  const int64_t gOutChannelGroup = gOutChannel  / group;

  const float * restrict pIn      = pDataIn;
  const float * restrict pGOut    = pDataGradOut;
  float * restrict const pGKernel = pDataGradKernel;

  const int gOutPixels= gOutHeight*gOutWidth ;

#ifndef VEDNN_USE_OPENMP
  const int64_t beginOChannel = 0 ;
  const int64_t nOChannel     = gOutChannelGroup ;
#endif
  {
    const int64_t nY = VLEN / gOutWidth ;

    _ve_lvl(nY*gOutWidth) ;
    __vr vrseq = _ve_vseq_v() ;			// xy
    __vr vry  = _ve_vdivsl_vvs(vrseq, gOutWidth) ;
    __vr vrx  = _ve_vsubsl_vvv(vrseq, _ve_vmulul_vsv(gOutWidth,vry)) ;
    __vr vri  = _ve_vmulsl_vsv(strideHeight, vry) ;
    __vr vrj  = _ve_vmulsl_vsv(strideWidth,  vrx) ;


    {
      for (int64_t g = 0; g < group; g++) {
	int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
	int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

	int64_t k=0;
	if ( (nOChannel & 0x01) == 1 ) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    for (int64_t r=0; r<gKernHeight; r++) {
	      for (int64_t s=0; s<gKernWidth; s++) {
		const int64_t kernelIndex = kernGroupOffset + ((k * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

		_ve_lvl(VLEN) ;
		__vr vrsum = _ve_vbrdu_vs_f32(0.0f) ;

		for (int64_t n=0; n<batch; n++) {
		  for (int64_t y=0; y<gOutHeight; y+=nY) {

		    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
		    const int64_t gop = y * gOutWidth ;

		    _ve_lvl(vl) ;

		    __vr vrh = _ve_vaddsl_vsv(r*dilationHeight-padHeight+y*strideHeight, vri) ;
		    __vr vrw = _ve_vaddsl_vsv(s*dilationWidth-padWidth,  vrj) ;

		    __vm256 vmh0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		    __vm256 vmh1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		    __vm256 vmw0 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		    __vm256 vmw1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		    __vm256 vmh  = _ve_andm_mmm(vmh0, vmh1) ;
		    __vm256 vmw  = _ve_andm_mmm(vmw0, vmw1) ;
		    __vm256 vmall = _ve_andm_mmm(vmh, vmw) ;

		    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		    const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k) * gOutHeight ) * gOutWidth + gop;
		    __vr vrpin = _ve_vsfa_vvss(_ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)),
					       2,
					       (uint64_t)pInChannel) ;

		    __vr vrin = _ve_vgtu_vvm(vrpin, vmall) ;
		    vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		    __vr vrgout = _ve_vldu_vss(4, pGOut+gOutIndex) ;

		    vrsum = _ve_vfmads_vvvv(vrsum, vrin, vrgout) ;

		  } // gOutPixels
		} // batch

		_ve_lvl(VLEN) ;
		vrsum = _ve_vfsums_vv(vrsum) ;
		float sum = _ve_lvs_svs_f32(vrsum,0) ;
		pGKernel[kernelIndex] = sum ;
	      } // kernWidth
	    } // kernHeight
	  } // inChannel
	  k++ ;
	}
	if ( ((nOChannel >> 1) & 0x01) == 1 ) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    for (int64_t r=0; r<gKernHeight; r++) {
	      for (int64_t s=0; s<gKernWidth; s++) {
		const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
		const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

		_ve_lvl(VLEN) ;
		__vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;

		for (int64_t n=0; n<batch; n++) {
		  for (int64_t y=0; y<gOutHeight; y+=nY) {

		    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
		    const int64_t gop = y * gOutWidth ;

		    _ve_lvl(vl) ;

		    __vr vrh = _ve_vaddsl_vsv(r*dilationHeight-padHeight+y*strideHeight, vri) ;
		    __vr vrw = _ve_vaddsl_vsv(s*dilationWidth-padWidth,  vrj) ;

		    __vm256 vmh0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		    __vm256 vmh1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		    __vm256 vmw0 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		    __vm256 vmw1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		    __vm256 vmh  = _ve_andm_mmm(vmh0, vmh1) ;
		    __vm256 vmw  = _ve_andm_mmm(vmw0, vmw1) ;
		    __vm256 vmall = _ve_andm_mmm(vmh, vmw) ;

		    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		    const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
		    const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;

		    __vr vrpin = _ve_vsfa_vvss(_ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)),
					       2,
					       (uint64_t)pInChannel) ;

		    __vr vrin = _ve_vgtu_vvm(vrpin, vmall) ;
		    vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		    __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		    __vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
		    __vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;

		    __vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;

		    vrsum01 = _ve_pvfmad_vvvv(vrsum01, vrinP, vrgout01) ;

		  } // gOutPixels
		} // batch

		_ve_lvl(VLEN) ;
		__vr vrsum0 = _ve_vfsums_vv(vrsum01) ;
		__vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01,32));

		pGKernel[kernelIndex0] = _ve_lvs_svs_f32(vrsum0,0) ;
		pGKernel[kernelIndex1] = _ve_lvs_svs_f32(vrsum1,0) ;

	      } // kernWidth
	    } // kernHeight
	  } // inChannel
	  k+=2;
	}
	if ( ((nOChannel >> 2) & 0x01) == 1 ) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    for (int64_t r=0; r<gKernHeight; r++) {
	      for (int64_t s=0; s<gKernWidth; s++) {
		const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
		const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
		const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
		const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

		_ve_lvl(VLEN) ;
		__vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;
		__vr vrsum23 = _ve_vbrd_vs_i64(0UL) ;

		for (int64_t n=0; n<batch; n++) {
		  for (int64_t y=0; y<gOutHeight; y+=nY) {

		    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
		    const int64_t gop = y * gOutWidth ;

		    _ve_lvl(vl) ;

		    __vr vrh = _ve_vaddsl_vsv(r*dilationHeight-padHeight+y*strideHeight, vri) ;
		    __vr vrw = _ve_vaddsl_vsv(s*dilationWidth-padWidth,  vrj) ;

		    __vm256 vmh0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		    __vm256 vmh1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		    __vm256 vmw0 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		    __vm256 vmw1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		    __vm256 vmh  = _ve_andm_mmm(vmh0, vmh1) ;
		    __vm256 vmw  = _ve_andm_mmm(vmw0, vmw1) ;
		    __vm256 vmall = _ve_andm_mmm(vmh, vmw) ;

		    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		    const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
		    const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;
		    const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop;
		    const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop;

		    __vr vrpin = _ve_vsfa_vvss(_ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)),
					       2,
					       (uint64_t)pInChannel) ;

		    __vr vrin = _ve_vgtu_vvm(vrpin, vmall) ;
		    vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		    __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		    __vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
		    __vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
		    __vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
		    __vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;

		    __vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
		    __vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;

		    vrsum01 = _ve_pvfmad_vvvv(vrsum01, vrinP, vrgout01) ;
		    vrsum23 = _ve_pvfmad_vvvv(vrsum23, vrinP, vrgout23) ;

		  } // gOutPixels
		} // batch

		_ve_lvl(VLEN) ;
		__vr vrsum0 = _ve_vfsums_vv(vrsum01) ;
		__vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01,32));
		__vr vrsum2 = _ve_vfsums_vv(vrsum23) ;
		__vr vrsum3 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23,32));


		pGKernel[kernelIndex0] = _ve_lvs_svs_f32(vrsum0,0) ;
		pGKernel[kernelIndex1] = _ve_lvs_svs_f32(vrsum1,0) ;
		pGKernel[kernelIndex2] = _ve_lvs_svs_f32(vrsum2,0) ;
		pGKernel[kernelIndex3] = _ve_lvs_svs_f32(vrsum3,0) ;

	      } // kernWidth
	    } // kernHeight
	  } // inChannel
	  k+=4;
	}
	for ( ;k<nOChannel; k+=8) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    for (int64_t r=0; r<gKernHeight; r++) {
	      for (int64_t s=0; s<gKernWidth; s++) {
		const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
		const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
		const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
		const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
		const int64_t kernelIndex4 = kernGroupOffset + (((k+4) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
		const int64_t kernelIndex5 = kernGroupOffset + (((k+5) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
		const int64_t kernelIndex6 = kernGroupOffset + (((k+6) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;
		const int64_t kernelIndex7 = kernGroupOffset + (((k+7) * inChannelGroup + c) * gKernHeight + r) * gKernWidth + s;

		_ve_lvl(VLEN) ;
		__vr vrsum01 = _ve_vbrd_vs_i64(0UL) ;
		__vr vrsum23 = _ve_vbrd_vs_i64(0UL) ;
		__vr vrsum45 = _ve_vbrd_vs_i64(0UL) ;
		__vr vrsum67 = _ve_vbrd_vs_i64(0UL) ;

		for (int64_t n=0; n<batch; n++) {
		  for (int64_t y=0; y<gOutHeight; y+=nY) {

		    const int64_t vl = gOutWidth * (gOutHeight - y < nY ? gOutHeight - y : nY) ;
		    const int64_t gop = y * gOutWidth ;

		    _ve_lvl(vl) ;

		    __vr vrh = _ve_vaddsl_vsv(r*dilationHeight-padHeight+y*strideHeight, vri) ;
		    __vr vrw = _ve_vaddsl_vsv(s*dilationWidth-padWidth,  vrj) ;

		    __vm256 vmh0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		    __vm256 vmh1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		    __vm256 vmw0 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		    __vm256 vmw1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		    __vm256 vmh  = _ve_andm_mmm(vmh0, vmh1) ;
		    __vm256 vmw  = _ve_andm_mmm(vmw0, vmw1) ;
		    __vm256 vmall = _ve_andm_mmm(vmh, vmw) ;

		    const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		    const int64_t gOutIndex0  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth + gop;
		    const int64_t gOutIndex1  = outGroupOffset + ((n * gOutChannel + k+1) * gOutHeight ) * gOutWidth + gop;
		    const int64_t gOutIndex2  = outGroupOffset + ((n * gOutChannel + k+2) * gOutHeight ) * gOutWidth + gop;
		    const int64_t gOutIndex3  = outGroupOffset + ((n * gOutChannel + k+3) * gOutHeight ) * gOutWidth + gop;
		    const int64_t gOutIndex4  = outGroupOffset + ((n * gOutChannel + k+4) * gOutHeight ) * gOutWidth + gop;
		    const int64_t gOutIndex5  = outGroupOffset + ((n * gOutChannel + k+5) * gOutHeight ) * gOutWidth + gop;
		    const int64_t gOutIndex6  = outGroupOffset + ((n * gOutChannel + k+6) * gOutHeight ) * gOutWidth + gop;
		    const int64_t gOutIndex7  = outGroupOffset + ((n * gOutChannel + k+7) * gOutHeight ) * gOutWidth + gop;

		    __vr vrpin = _ve_vsfa_vvss(_ve_vaddul_vvv(vrw, _ve_vmulul_vsv(inWidth,vrh)),
					       2,
					       (uint64_t)pInChannel) ;

		    __vr vrin = _ve_vgtu_vvm(vrpin, vmall) ;
		    vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		    __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		    __vr vrgout0 = _ve_vldu_vss(4, pGOut+gOutIndex0) ;
		    __vr vrgout1 = _ve_vldu_vss(4, pGOut+gOutIndex1) ;
		    __vr vrgout2 = _ve_vldu_vss(4, pGOut+gOutIndex2) ;
		    __vr vrgout3 = _ve_vldu_vss(4, pGOut+gOutIndex3) ;
		    __vr vrgout4 = _ve_vldu_vss(4, pGOut+gOutIndex4) ;
		    __vr vrgout5 = _ve_vldu_vss(4, pGOut+gOutIndex5) ;
		    __vr vrgout6 = _ve_vldu_vss(4, pGOut+gOutIndex6) ;
		    __vr vrgout7 = _ve_vldu_vss(4, pGOut+gOutIndex7) ;

		    __vr vrgout01 = _ve_vshf_vvvs(vrgout0, vrgout1, VE_VSHUFFLE_YUZU) ;
		    __vr vrgout23 = _ve_vshf_vvvs(vrgout2, vrgout3, VE_VSHUFFLE_YUZU) ;
		    __vr vrgout45 = _ve_vshf_vvvs(vrgout4, vrgout5, VE_VSHUFFLE_YUZU) ;
		    __vr vrgout67 = _ve_vshf_vvvs(vrgout6, vrgout7, VE_VSHUFFLE_YUZU) ;

		    vrsum01 = _ve_pvfmad_vvvv(vrsum01, vrinP, vrgout01) ;
		    vrsum23 = _ve_pvfmad_vvvv(vrsum23, vrinP, vrgout23) ;
		    vrsum45 = _ve_pvfmad_vvvv(vrsum45, vrinP, vrgout45) ;
		    vrsum67 = _ve_pvfmad_vvvv(vrsum67, vrinP, vrgout67) ;

		  } // gOutPixels
		} // batch

		_ve_lvl(VLEN) ;
		__vr vrsum0 = _ve_vfsums_vv(vrsum01) ;
		__vr vrsum1 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum01,32));
		__vr vrsum2 = _ve_vfsums_vv(vrsum23) ;
		__vr vrsum3 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum23,32));
		__vr vrsum4 = _ve_vfsums_vv(vrsum45) ;
		__vr vrsum5 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum45,32));
		__vr vrsum6 = _ve_vfsums_vv(vrsum67) ;
		__vr vrsum7 = _ve_vfsums_vv(_ve_vsll_vvs(vrsum67,32));

		pGKernel[kernelIndex0] = _ve_lvs_svs_f32(vrsum0,0) ;
		pGKernel[kernelIndex1] = _ve_lvs_svs_f32(vrsum1,0) ;
		pGKernel[kernelIndex2] = _ve_lvs_svs_f32(vrsum2,0) ;
		pGKernel[kernelIndex3] = _ve_lvs_svs_f32(vrsum3,0) ;
		pGKernel[kernelIndex4] = _ve_lvs_svs_f32(vrsum4,0) ;
		pGKernel[kernelIndex5] = _ve_lvs_svs_f32(vrsum5,0) ;
		pGKernel[kernelIndex6] = _ve_lvs_svs_f32(vrsum6,0) ;
		pGKernel[kernelIndex7] = _ve_lvs_svs_f32(vrsum7,0) ;

	      } // kernWidth
	    } // kernHeight
	  } // inChannel
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}
