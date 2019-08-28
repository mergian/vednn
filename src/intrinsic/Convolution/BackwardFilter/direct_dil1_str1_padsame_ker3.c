#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "velintrin.h"
#define VLEN	(256)

vednnError_t
vednnConvolutionBackwardFilter_direct_dil1_str1_padsame_ker3(
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
  const int64_t gKernWidth  = pParamGradKernel->width;		/* must be 3 */
  const int64_t gKernHeight = pParamGradKernel->height;		/* must be 3 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
//  const int64_t padWidth       = pParamConv->padWidth;	/* must be 1 */
//  const int64_t padHeight      = pParamConv->padHeight;	/* must be 1 */
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

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
    {
      for (int64_t g = 0; g < group; g++) {
	int64_t inGroupOffset   = g * inChannelGroup  * inHeight  * inWidth;
	int64_t outGroupOffset  = (g * gOutChannelGroup + beginOChannel) * gOutHeight * gOutWidth;
	int64_t kernGroupOffset = (g * gOutChannelGroup + beginOChannel) * inChannelGroup * gKernHeight * gKernWidth;

	int64_t k = 0 ;
	if ( (nOChannel & 0x01) == 1 ) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;

	    __vr vrsum_r0s0 = _vel_vbrds_vsl(0.0f, VLEN) ;
	    __vr vrsum_r0s1 = _vel_vbrds_vsl(0.0f, VLEN) ;
	    __vr vrsum_r0s2 = _vel_vbrds_vsl(0.0f, VLEN) ;
	    __vr vrsum_r1s0 = _vel_vbrds_vsl(0.0f, VLEN) ;
	    __vr vrsum_r1s1 = _vel_vbrds_vsl(0.0f, VLEN) ;
	    __vr vrsum_r1s2 = _vel_vbrds_vsl(0.0f, VLEN) ;
	    __vr vrsum_r2s0 = _vel_vbrds_vsl(0.0f, VLEN) ;
	    __vr vrsum_r2s1 = _vel_vbrds_vsl(0.0f, VLEN) ;
	    __vr vrsum_r2s2 = _vel_vbrds_vsl(0.0f, VLEN) ;

	    for (int64_t n=0; n<batch; n++) {
	      for (int64_t x=0; x<gOutWidth; x+=VLEN) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

		const int64_t vl = gOutWidth - x < VLEN ? gOutWidth - x : VLEN ;

		__vr vrx   = _vel_vaddsl_vsvl(x, _vel_vseq_vl(vl), vl) ;	// x

		__vm256 vmw0_s0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrx, vl), vl) ;		// condition(  1<=x)
		__vm256 vmw1_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth-1,vrx, vl), vl) ;	// condition(x+1< inWidth)

		__vr vrin_r1s0, vrin_r1s1, vrin_r1s2 ;
		__vr vrin_r2s0, vrin_r2s1, vrin_r2s2 ;

		int64_t y=0 ;
		{
		  const int64_t gop = y * gOutWidth + x ;

		  /* memory access errors mihgt be caused (vrin) */
		  vrin_r1s0    = _vel_vldu_vssl(4,&pInChannel[gop        -1], vl) ;
		  vrin_r1s1    = _vel_vldu_vssl(4,&pInChannel[gop          ], vl) ;
		  vrin_r1s2    = _vel_vldu_vssl(4,&pInChannel[gop        +1], vl) ;
		  vrin_r2s0    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth-1], vl) ;
		  vrin_r2s1    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth  ], vl) ;
		  vrin_r2s2    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth+1], vl) ;

		  __vr vrgout = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+0*gOutPixels, vl) ;

		  vrin_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s0, vmw0_s0, vl) ;
		  vrsum_r1s0 = _vel_vfmads_vvvvvl(vrsum_r1s0, vrin_r1s0, vrgout, vrsum_r1s0, vl) ;

		  // vrin_r1s1 ; // no need to use mask
		  vrsum_r1s1 = _vel_vfmads_vvvvvl(vrsum_r1s1, vrin_r1s1, vrgout, vrsum_r1s1, vl) ;

		  vrin_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s2, vmw1_s2, vl) ;
		  vrsum_r1s2 = _vel_vfmads_vvvvvl(vrsum_r1s2, vrin_r1s2, vrgout, vrsum_r1s2, vl) ;

		  vrin_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s0, vmw0_s0, vl) ;
		  vrsum_r2s0 = _vel_vfmads_vvvvvl(vrsum_r2s0, vrin_r2s0, vrgout, vrsum_r2s0, vl) ;

		  // vrin_r2s1 ; // no need to use mask
		  vrsum_r2s1 = _vel_vfmads_vvvvvl(vrsum_r2s1, vrin_r2s1, vrgout, vrsum_r2s1, vl) ;

		  vrin_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s2, vmw1_s2, vl) ;
		  vrsum_r2s2 = _vel_vfmads_vvvvvl(vrsum_r2s2, vrin_r2s2, vrgout, vrsum_r2s2, vl) ;
		}
		for (y=1; y<gOutHeight-1; y++)
		{
		  const int64_t gop = y * gOutWidth + x ;

		  /* memory access errors mihgt be caused (vrin) */
		  __vr vrin_r0s0    = vrin_r1s0 ;
		  __vr vrin_r0s1    = vrin_r1s1 ;
		  __vr vrin_r0s2    = vrin_r1s2 ;
		       vrin_r1s0    = vrin_r2s0 ;
		       vrin_r1s1    = vrin_r2s1 ;
		       vrin_r1s2    = vrin_r2s2 ;
		       vrin_r2s0    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth-1], vl) ;
		       vrin_r2s1    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth  ], vl) ;
		       vrin_r2s2    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth+1], vl) ;

		  __vr vrgout = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+0*gOutPixels, vl) ;

		  vrsum_r0s0 = _vel_vfmads_vvvvvl(vrsum_r0s0, vrin_r0s0, vrgout, vrsum_r0s0, vl) ;
		  vrsum_r0s1 = _vel_vfmads_vvvvvl(vrsum_r0s1, vrin_r0s1, vrgout, vrsum_r0s1, vl) ;
		  vrsum_r0s2 = _vel_vfmads_vvvvvl(vrsum_r0s2, vrin_r0s2, vrgout, vrsum_r0s2, vl) ;

		  vrsum_r1s0 = _vel_vfmads_vvvvvl(vrsum_r1s0, vrin_r1s0, vrgout, vrsum_r1s0, vl) ;
		  vrsum_r1s1 = _vel_vfmads_vvvvvl(vrsum_r1s1, vrin_r1s1, vrgout, vrsum_r1s1, vl) ;
		  vrsum_r1s2 = _vel_vfmads_vvvvvl(vrsum_r1s2, vrin_r1s2, vrgout, vrsum_r1s2, vl) ;

		  vrin_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s0, vmw0_s0, vl) ;
		  vrsum_r2s0 = _vel_vfmads_vvvvvl(vrsum_r2s0, vrin_r2s0, vrgout, vrsum_r2s0, vl) ;

		  // vrin_r2s1 ; // no need to use mask
		  vrsum_r2s1 = _vel_vfmads_vvvvvl(vrsum_r2s1, vrin_r2s1, vrgout, vrsum_r2s1, vl) ;

		  vrin_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s2, vmw1_s2, vl) ;
		  vrsum_r2s2 = _vel_vfmads_vvvvvl(vrsum_r2s2, vrin_r2s2, vrgout, vrsum_r2s2, vl) ;
		}
		{
		  const int64_t gop = y * gOutWidth + x ;

		  /* memory access errors mihgt be caused (vrin) */
		  __vr vrin_r0s0    = vrin_r1s0 ;
		  __vr vrin_r0s1    = vrin_r1s1 ;
		  __vr vrin_r0s2    = vrin_r1s2 ;
		       vrin_r1s0    = vrin_r2s0 ;
		       vrin_r1s1    = vrin_r2s1 ;
		       vrin_r1s2    = vrin_r2s2 ;

		  __vr vrgout = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+0*gOutPixels, vl) ;

		  vrsum_r0s0 = _vel_vfmads_vvvvvl(vrsum_r0s0, vrin_r0s0, vrgout, vrsum_r0s0, vl) ;
		  vrsum_r0s1 = _vel_vfmads_vvvvvl(vrsum_r0s1, vrin_r0s1, vrgout, vrsum_r0s1, vl) ;
		  vrsum_r0s2 = _vel_vfmads_vvvvvl(vrsum_r0s2, vrin_r0s2, vrgout, vrsum_r0s2, vl) ;

		  vrsum_r1s0 = _vel_vfmads_vvvvvl(vrsum_r1s0, vrin_r1s0, vrgout, vrsum_r1s0, vl) ;
		  vrsum_r1s1 = _vel_vfmads_vvvvvl(vrsum_r1s1, vrin_r1s1, vrgout, vrsum_r1s1, vl) ;
		  vrsum_r1s2 = _vel_vfmads_vvvvvl(vrsum_r1s2, vrin_r1s2, vrgout, vrsum_r1s2, vl) ;

		}
	      } // gOutWidth
	    } // batch

	    vrsum_r0s0 = _vel_vfsums_vvl(vrsum_r0s0, VLEN) ;
	    vrsum_r0s1 = _vel_vfsums_vvl(vrsum_r0s1, VLEN) ;
	    vrsum_r0s2 = _vel_vfsums_vvl(vrsum_r0s2, VLEN) ;
	    vrsum_r1s0 = _vel_vfsums_vvl(vrsum_r1s0, VLEN) ;
	    vrsum_r1s1 = _vel_vfsums_vvl(vrsum_r1s1, VLEN) ;
	    vrsum_r1s2 = _vel_vfsums_vvl(vrsum_r1s2, VLEN) ;
	    vrsum_r2s0 = _vel_vfsums_vvl(vrsum_r2s0, VLEN) ;
	    vrsum_r2s1 = _vel_vfsums_vvl(vrsum_r2s1, VLEN) ;
	    vrsum_r2s2 = _vel_vfsums_vvl(vrsum_r2s2, VLEN) ;

	    _vel_vstu_vssl(vrsum_r0s0,4,pGKernel+kernelIndex+0, 1) ;
	    _vel_vstu_vssl(vrsum_r0s1,4,pGKernel+kernelIndex+1, 1) ;
	    _vel_vstu_vssl(vrsum_r0s2,4,pGKernel+kernelIndex+2, 1) ;
	    _vel_vstu_vssl(vrsum_r1s0,4,pGKernel+kernelIndex+3, 1) ;
	    _vel_vstu_vssl(vrsum_r1s1,4,pGKernel+kernelIndex+4, 1) ;
	    _vel_vstu_vssl(vrsum_r1s2,4,pGKernel+kernelIndex+5, 1) ;
	    _vel_vstu_vssl(vrsum_r2s0,4,pGKernel+kernelIndex+6, 1) ;
	    _vel_vstu_vssl(vrsum_r2s1,4,pGKernel+kernelIndex+7, 1) ;
	    _vel_vstu_vssl(vrsum_r2s2,4,pGKernel+kernelIndex+8, 1) ;
	  } // inChannel


	  k+=1;
	}
	if ( ((nOChannel >> 1) & 0x01) == 1 ) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;

#define INIT_VRSUM_2(TOKEN, INDEX)	\
		__vr vrsum01_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

	    INIT_VRSUM_2(r0s0, 0) ;
	    INIT_VRSUM_2(r0s1, 1) ;
	    INIT_VRSUM_2(r0s2, 2) ;
	    INIT_VRSUM_2(r1s0, 3) ;
	    INIT_VRSUM_2(r1s1, 4) ;
	    INIT_VRSUM_2(r1s2, 5) ;
	    INIT_VRSUM_2(r2s0, 6) ;
	    INIT_VRSUM_2(r2s1, 7) ;
	    INIT_VRSUM_2(r2s2, 8) ;
#undef INIT_VRSUM_2

	    for (int64_t n=0; n<batch; n++)  {
	      for (int64_t x=0; x<gOutWidth; x+=VLEN) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

		const int64_t vl = gOutWidth - x < VLEN ? gOutWidth - x : VLEN ;

		__vr vrx = _vel_vaddsl_vsvl(x, _vel_vseq_vl(vl), vl) ;	// op + xy

		__vm256 vmw0_s0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrx, vl), vl) ;		// condition(  1<=x)
		__vm256 vmw1_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth-1,vrx, vl), vl) ;	// condition(x+1< inWidth)

		__vr vrinP_r1s0, vrinP_r1s1, vrinP_r1s2 ;
		__vr vrinP_r2s0, vrinP_r2s1, vrinP_r2s2 ;

		int64_t y=0 ;
		{
		  const int64_t gop = y * gOutWidth + x ;

		  /* memory access errors mihgt be caused (vrin) */
		  __vr vrin_r1s0    = _vel_vldu_vssl(4,&pInChannel[gop        -1], vl) ;
		  __vr vrin_r1s1    = _vel_vldu_vssl(4,&pInChannel[gop          ], vl) ;
		  __vr vrin_r1s2    = _vel_vldu_vssl(4,&pInChannel[gop        +1], vl) ;
		  __vr vrin_r2s0    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth-1], vl) ;
		  __vr vrin_r2s1    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth  ], vl) ;
		  __vr vrin_r2s2    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth+1], vl) ;

		  __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+0*gOutPixels, vl) ;
		  __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+1*gOutPixels, vl) ;

		  __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;

		  vrin_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s0, vmw0_s0, vl) ;
		  vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl) ;

		  // vrin_r1s1 ; // no need to use mask
		  vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl) ;

		  vrin_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s2, vmw1_s2, vl) ;
		  vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl) ;

		  vrin_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s0, vmw0_s0, vl) ;
		  vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s0 = _vel_pvfmad_vvvvvl(vrsum01_r2s0, vrinP_r2s0, vrgout01, vrsum01_r2s0, vl) ;

		  // vrin_r2s1 ; // no need to use mask
		  vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s1 = _vel_pvfmad_vvvvvl(vrsum01_r2s1, vrinP_r2s1, vrgout01, vrsum01_r2s1, vl) ;

		  vrin_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s2, vmw1_s2, vl) ;
		  vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP_r2s2, vrgout01, vrsum01_r2s2, vl) ;
		}
		for (y=1; y<gOutHeight-1; y++) {

		  const int64_t gop = y * gOutWidth + x ;

		  /* memory access errors mihgt be caused (vrin) */
		  __vr vrin_r2s0    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth-1], vl) ;
		  __vr vrin_r2s1    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth  ], vl) ;
		  __vr vrin_r2s2    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth+1], vl) ;

		  __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+0*gOutPixels, vl) ;
		  __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+1*gOutPixels, vl) ;

		  __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;

		  __vr vrinP_r0s0 = vrinP_r1s0 ;
		  vrsum01_r0s0 = _vel_pvfmad_vvvvvl(vrsum01_r0s0, vrinP_r0s0, vrgout01, vrsum01_r0s0, vl) ;

		  __vr vrinP_r0s1 = vrinP_r1s1 ;
		  vrsum01_r0s1 = _vel_pvfmad_vvvvvl(vrsum01_r0s1, vrinP_r0s1, vrgout01, vrsum01_r0s1, vl) ;

		  __vr vrinP_r0s2 = vrinP_r1s2 ;
		  vrsum01_r0s2 = _vel_pvfmad_vvvvvl(vrsum01_r0s2, vrinP_r0s2, vrgout01, vrsum01_r0s2, vl) ;

		  vrinP_r1s0 = vrinP_r2s0 ;
		  vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl) ;

		  vrinP_r1s1 = vrinP_r2s1 ;
		  vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl) ;

		  vrinP_r1s2 = vrinP_r2s2 ;
		  vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl) ;

		  vrin_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s0, vmw0_s0, vl) ;
		  vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s0 = _vel_pvfmad_vvvvvl(vrsum01_r2s0, vrinP_r2s0, vrgout01, vrsum01_r2s0, vl) ;

		  // vrin_r2s1 ; // no need to use mask
		  vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s1 = _vel_pvfmad_vvvvvl(vrsum01_r2s1, vrinP_r2s1, vrgout01, vrsum01_r2s1, vl) ;

		  vrin_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s2, vmw1_s2, vl) ;
		  vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP_r2s2, vrgout01, vrsum01_r2s2, vl) ;
		}
		{
		  const int64_t gop = y * gOutWidth + x ;

		  __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+0*gOutPixels, vl) ;
		  __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+1*gOutPixels, vl) ;

		  __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;

		  __vr vrinP_r0s0 = vrinP_r1s0 ;
		  vrsum01_r0s0 = _vel_pvfmad_vvvvvl(vrsum01_r0s0, vrinP_r0s0, vrgout01, vrsum01_r0s0, vl) ;

		  __vr vrinP_r0s1 = vrinP_r1s1 ;
		  vrsum01_r0s1 = _vel_pvfmad_vvvvvl(vrsum01_r0s1, vrinP_r0s1, vrgout01, vrsum01_r0s1, vl) ;

		  __vr vrinP_r0s2 = vrinP_r1s2 ;
		  vrsum01_r0s2 = _vel_pvfmad_vvvvvl(vrsum01_r0s2, vrinP_r0s2, vrgout01, vrsum01_r0s2, vl) ;

		  vrinP_r1s0 = vrinP_r2s0 ;
		  vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl) ;

		  vrinP_r1s1 = vrinP_r2s1 ;
		  vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl) ;

		  vrinP_r1s2 = vrinP_r2s2 ;
		  vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl) ;
		}
	      } // gOutWidth

	    } // batch

#define VSUM_STORE_3X3_UPPER(VRSUMTOKEN, KERNELINDEX)		\
{								\
  __vr vrsumU_r0s0 = _vel_vfsums_vvl(VRSUMTOKEN ## _r0s0, VLEN) ;	\
  __vr vrsumU_r0s1 = _vel_vfsums_vvl(VRSUMTOKEN ## _r0s1, VLEN) ;	\
  __vr vrsumU_r0s2 = _vel_vfsums_vvl(VRSUMTOKEN ## _r0s2, VLEN) ;	\
  __vr vrsumU_r1s0 = _vel_vfsums_vvl(VRSUMTOKEN ## _r1s0, VLEN) ;	\
  __vr vrsumU_r1s1 = _vel_vfsums_vvl(VRSUMTOKEN ## _r1s1, VLEN) ;	\
  __vr vrsumU_r1s2 = _vel_vfsums_vvl(VRSUMTOKEN ## _r1s2, VLEN) ;	\
  __vr vrsumU_r2s0 = _vel_vfsums_vvl(VRSUMTOKEN ## _r2s0, VLEN) ;	\
  __vr vrsumU_r2s1 = _vel_vfsums_vvl(VRSUMTOKEN ## _r2s1, VLEN) ;	\
  __vr vrsumU_r2s2 = _vel_vfsums_vvl(VRSUMTOKEN ## _r2s2, VLEN) ;	\
  _vel_vstu_vssl(vrsumU_r0s0,4,pGKernel+(KERNELINDEX)+0, 1) ;	\
  _vel_vstu_vssl(vrsumU_r0s1,4,pGKernel+(KERNELINDEX)+1, 1) ;	\
  _vel_vstu_vssl(vrsumU_r0s2,4,pGKernel+(KERNELINDEX)+2, 1) ;	\
  _vel_vstu_vssl(vrsumU_r1s0,4,pGKernel+(KERNELINDEX)+3, 1) ;	\
  _vel_vstu_vssl(vrsumU_r1s1,4,pGKernel+(KERNELINDEX)+4, 1) ;	\
  _vel_vstu_vssl(vrsumU_r1s2,4,pGKernel+(KERNELINDEX)+5, 1) ;	\
  _vel_vstu_vssl(vrsumU_r2s0,4,pGKernel+(KERNELINDEX)+6, 1) ;	\
  _vel_vstu_vssl(vrsumU_r2s1,4,pGKernel+(KERNELINDEX)+7, 1) ;	\
  _vel_vstu_vssl(vrsumU_r2s2,4,pGKernel+(KERNELINDEX)+8, 1) ;	\
  }
#define VSUM_STORE_3X3_LOWER(VRSUMTOKEN, KERNELINDEX)				\
  {										\
  __vr vrsumL_r0s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r0s0,32, VLEN), VLEN) ;	\
  __vr vrsumL_r0s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r0s1,32, VLEN), VLEN) ;	\
  __vr vrsumL_r0s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r0s2,32, VLEN), VLEN) ;	\
  __vr vrsumL_r1s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r1s0,32, VLEN), VLEN) ;	\
  __vr vrsumL_r1s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r1s1,32, VLEN), VLEN) ;	\
  __vr vrsumL_r1s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r1s2,32, VLEN), VLEN) ;	\
  __vr vrsumL_r2s0 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r2s0,32, VLEN), VLEN) ;	\
  __vr vrsumL_r2s1 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r2s1,32, VLEN), VLEN) ;	\
  __vr vrsumL_r2s2 = _vel_vfsums_vvl(_vel_vsll_vvsl(VRSUMTOKEN ## _r2s2,32, VLEN), VLEN) ;	\
  _vel_vstu_vssl(vrsumL_r0s0,4,pGKernel+(KERNELINDEX)+0, 1) ;			\
  _vel_vstu_vssl(vrsumL_r0s1,4,pGKernel+(KERNELINDEX)+1, 1) ;			\
  _vel_vstu_vssl(vrsumL_r0s2,4,pGKernel+(KERNELINDEX)+2, 1) ;			\
  _vel_vstu_vssl(vrsumL_r1s0,4,pGKernel+(KERNELINDEX)+3, 1) ;			\
  _vel_vstu_vssl(vrsumL_r1s1,4,pGKernel+(KERNELINDEX)+4, 1) ;			\
  _vel_vstu_vssl(vrsumL_r1s2,4,pGKernel+(KERNELINDEX)+5, 1) ;			\
  _vel_vstu_vssl(vrsumL_r2s0,4,pGKernel+(KERNELINDEX)+6, 1) ;			\
  _vel_vstu_vssl(vrsumL_r2s1,4,pGKernel+(KERNELINDEX)+7, 1) ;			\
  _vel_vstu_vssl(vrsumL_r2s2,4,pGKernel+(KERNELINDEX)+8, 1) ;			\
}

	  VSUM_STORE_3X3_UPPER(vrsum01, kernelIndex0) ;
	  VSUM_STORE_3X3_LOWER(vrsum01, kernelIndex1) ;

	  } // inChannel
	  k+=2;
	}
	if ( ((nOChannel >> 2) & 0x01) == 1 ) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight) * gKernWidth;

#define INIT_VRSUM_4(TOKEN, INDEX)	\
  __vr vrsum01_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum23_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

	    INIT_VRSUM_4(r0s0, 0) ;
	    INIT_VRSUM_4(r0s1, 1) ;
	    INIT_VRSUM_4(r0s2, 2) ;
	    INIT_VRSUM_4(r1s0, 3) ;
	    INIT_VRSUM_4(r1s1, 4) ;
	    INIT_VRSUM_4(r1s2, 5) ;
	    INIT_VRSUM_4(r2s0, 6) ;
	    INIT_VRSUM_4(r2s1, 7) ;
	    INIT_VRSUM_4(r2s2, 8) ;
#undef INIT_VRSUM_4

	    for (int64_t n=0; n<batch; n++) {
	      for (int64_t x=0; x<gOutWidth; x+=VLEN) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

		const int64_t vl = gOutWidth - x < VLEN ? gOutWidth - x : VLEN ;

		__vr vrx = _vel_vaddsl_vsvl(x, _vel_vseq_vl(vl), vl) ;	// op + xy

		__vm256 vmw0_s0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrx, vl), vl) ;		// condition(  1<=x)
		__vm256 vmw1_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth-1,vrx, vl), vl) ;	// condition(x+1< inWidth)

		__vr vrinP_r1s0, vrinP_r1s1, vrinP_r1s2 ;
		__vr vrinP_r2s0, vrinP_r2s1, vrinP_r2s2 ;

		int64_t y=0 ;
		{
		  const int64_t gop = y * gOutWidth + x ;

		  /* memory access errors mihgt be caused (vrin) */
		  __vr vrin_r1s0    = _vel_vldu_vssl(4,&pInChannel[gop        -1], vl) ;
		  __vr vrin_r1s1    = _vel_vldu_vssl(4,&pInChannel[gop          ], vl) ;
		  __vr vrin_r1s2    = _vel_vldu_vssl(4,&pInChannel[gop        +1], vl) ;
		  __vr vrin_r2s0    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth-1], vl) ;
		  __vr vrin_r2s1    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth  ], vl) ;
		  __vr vrin_r2s2    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth+1], vl) ;

		  __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+0*gOutPixels, vl) ;
		  __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+1*gOutPixels, vl) ;
		  __vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+2*gOutPixels, vl) ;
		  __vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+3*gOutPixels, vl) ;

		  __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		  __vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;

		  vrin_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s0, vmw0_s0, vl) ;
		  vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl) ;
		  vrsum23_r1s0 = _vel_pvfmad_vvvvvl(vrsum23_r1s0, vrinP_r1s0, vrgout23, vrsum23_r1s0, vl) ;

		  // vrin_r1s1 ; // no need to use mask
		  vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl) ;
		  vrsum23_r1s1 = _vel_pvfmad_vvvvvl(vrsum23_r1s1, vrinP_r1s1, vrgout23, vrsum23_r1s1, vl) ;

		  vrin_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s2, vmw1_s2, vl) ;
		  vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl) ;
		  vrsum23_r1s2 = _vel_pvfmad_vvvvvl(vrsum23_r1s2, vrinP_r1s2, vrgout23, vrsum23_r1s2, vl) ;

		  vrin_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s0, vmw0_s0, vl) ;
		  vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s0 = _vel_pvfmad_vvvvvl(vrsum01_r2s0, vrinP_r2s0, vrgout01, vrsum01_r2s0, vl) ;
		  vrsum23_r2s0 = _vel_pvfmad_vvvvvl(vrsum23_r2s0, vrinP_r2s0, vrgout23, vrsum23_r2s0, vl) ;

		  // vrin_r2s1 ; // no need to use mask
		  vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s1 = _vel_pvfmad_vvvvvl(vrsum01_r2s1, vrinP_r2s1, vrgout01, vrsum01_r2s1, vl) ;
		  vrsum23_r2s1 = _vel_pvfmad_vvvvvl(vrsum23_r2s1, vrinP_r2s1, vrgout23, vrsum23_r2s1, vl) ;

		  vrin_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s2, vmw1_s2, vl) ;
		  vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP_r2s2, vrgout01, vrsum01_r2s2, vl) ;
		  vrsum23_r2s2 = _vel_pvfmad_vvvvvl(vrsum23_r2s2, vrinP_r2s2, vrgout23, vrsum23_r2s2, vl) ;
		}
		for (y=1; y<gOutHeight-1; y++) {

		  const int64_t gop = y * gOutWidth + x ;

		  /* memory access errors mihgt be caused (vrin) */
		  __vr vrin_r2s0    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth-1], vl) ;
		  __vr vrin_r2s1    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth  ], vl) ;
		  __vr vrin_r2s2    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth+1], vl) ;

		  __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+0*gOutPixels, vl) ;
		  __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+1*gOutPixels, vl) ;
		  __vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+2*gOutPixels, vl) ;
		  __vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+3*gOutPixels, vl) ;

		  __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		  __vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;

		  __vr vrinP_r0s0 = vrinP_r1s0 ;
		  vrsum01_r0s0 = _vel_pvfmad_vvvvvl(vrsum01_r0s0, vrinP_r0s0, vrgout01, vrsum01_r0s0, vl) ;
		  vrsum23_r0s0 = _vel_pvfmad_vvvvvl(vrsum23_r0s0, vrinP_r0s0, vrgout23, vrsum23_r0s0, vl) ;

		  __vr vrinP_r0s1 = vrinP_r1s1 ;
		  vrsum01_r0s1 = _vel_pvfmad_vvvvvl(vrsum01_r0s1, vrinP_r0s1, vrgout01, vrsum01_r0s1, vl) ;
		  vrsum23_r0s1 = _vel_pvfmad_vvvvvl(vrsum23_r0s1, vrinP_r0s1, vrgout23, vrsum23_r0s1, vl) ;

		  __vr vrinP_r0s2 = vrinP_r1s2 ;
		  vrsum01_r0s2 = _vel_pvfmad_vvvvvl(vrsum01_r0s2, vrinP_r0s2, vrgout01, vrsum01_r0s2, vl) ;
		  vrsum23_r0s2 = _vel_pvfmad_vvvvvl(vrsum23_r0s2, vrinP_r0s2, vrgout23, vrsum23_r0s2, vl) ;

		  vrinP_r1s0 = vrinP_r2s0 ;
		  vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl) ;
		  vrsum23_r1s0 = _vel_pvfmad_vvvvvl(vrsum23_r1s0, vrinP_r1s0, vrgout23, vrsum23_r1s0, vl) ;

		  vrinP_r1s1 = vrinP_r2s1 ;
		  vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl) ;
		  vrsum23_r1s1 = _vel_pvfmad_vvvvvl(vrsum23_r1s1, vrinP_r1s1, vrgout23, vrsum23_r1s1, vl) ;

		  vrinP_r1s2 = vrinP_r2s2 ;
		  vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl) ;
		  vrsum23_r1s2 = _vel_pvfmad_vvvvvl(vrsum23_r1s2, vrinP_r1s2, vrgout23, vrsum23_r1s2, vl) ;

		  vrin_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s0, vmw0_s0, vl) ;
		  vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s0 = _vel_pvfmad_vvvvvl(vrsum01_r2s0, vrinP_r2s0, vrgout01, vrsum01_r2s0, vl) ;
		  vrsum23_r2s0 = _vel_pvfmad_vvvvvl(vrsum23_r2s0, vrinP_r2s0, vrgout23, vrsum23_r2s0, vl) ;

		  // vrin_r2s1 ; // no need to use mask
		  vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s1 = _vel_pvfmad_vvvvvl(vrsum01_r2s1, vrinP_r2s1, vrgout01, vrsum01_r2s1, vl) ;
		  vrsum23_r2s1 = _vel_pvfmad_vvvvvl(vrsum23_r2s1, vrinP_r2s1, vrgout23, vrsum23_r2s1, vl) ;

		  vrin_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s2, vmw1_s2, vl) ;
		  vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP_r2s2, vrgout01, vrsum01_r2s2, vl) ;
		  vrsum23_r2s2 = _vel_pvfmad_vvvvvl(vrsum23_r2s2, vrinP_r2s2, vrgout23, vrsum23_r2s2, vl) ;
		}
		{
		  const int64_t gop = y * gOutWidth + x ;

		  __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+0*gOutPixels, vl) ;
		  __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+1*gOutPixels, vl) ;
		  __vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+2*gOutPixels, vl) ;
		  __vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+3*gOutPixels, vl) ;

		  __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		  __vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;

		  __vr vrinP_r0s0 = vrinP_r1s0 ;
		  vrsum01_r0s0 = _vel_pvfmad_vvvvvl(vrsum01_r0s0, vrinP_r0s0, vrgout01, vrsum01_r0s0, vl) ;
		  vrsum23_r0s0 = _vel_pvfmad_vvvvvl(vrsum23_r0s0, vrinP_r0s0, vrgout23, vrsum23_r0s0, vl) ;

		  __vr vrinP_r0s1 = vrinP_r1s1 ;
		  vrsum01_r0s1 = _vel_pvfmad_vvvvvl(vrsum01_r0s1, vrinP_r0s1, vrgout01, vrsum01_r0s1, vl) ;
		  vrsum23_r0s1 = _vel_pvfmad_vvvvvl(vrsum23_r0s1, vrinP_r0s1, vrgout23, vrsum23_r0s1, vl) ;

		  __vr vrinP_r0s2 = vrinP_r1s2 ;
		  vrsum01_r0s2 = _vel_pvfmad_vvvvvl(vrsum01_r0s2, vrinP_r0s2, vrgout01, vrsum01_r0s2, vl) ;
		  vrsum23_r0s2 = _vel_pvfmad_vvvvvl(vrsum23_r0s2, vrinP_r0s2, vrgout23, vrsum23_r0s2, vl) ;

		  vrinP_r1s0 = vrinP_r2s0 ;
		  vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl) ;
		  vrsum23_r1s0 = _vel_pvfmad_vvvvvl(vrsum23_r1s0, vrinP_r1s0, vrgout23, vrsum23_r1s0, vl) ;

		  vrinP_r1s1 = vrinP_r2s1 ;
		  vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl) ;
		  vrsum23_r1s1 = _vel_pvfmad_vvvvvl(vrsum23_r1s1, vrinP_r1s1, vrgout23, vrsum23_r1s1, vl) ;

		  vrinP_r1s2 = vrinP_r2s2 ;
		  vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl) ;
		  vrsum23_r1s2 = _vel_pvfmad_vvvvvl(vrsum23_r1s2, vrinP_r1s2, vrgout23, vrsum23_r1s2, vl) ;
		}
	      } // gOutWidth
	    } // batch

	    VSUM_STORE_3X3_UPPER(vrsum01, kernelIndex0) ;
	    VSUM_STORE_3X3_LOWER(vrsum01, kernelIndex1) ;
	    VSUM_STORE_3X3_UPPER(vrsum23, kernelIndex2) ;
	    VSUM_STORE_3X3_LOWER(vrsum23, kernelIndex3) ;

	  } // inChannel

	  k+=4;
	}
	for ( ;k<nOChannel; k+=8) {
	  for (int64_t c=0; c<inChannelGroup; c++) {
	    const int64_t kernelIndex0 = kernGroupOffset + ((k     * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex1 = kernGroupOffset + (((k+1) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex2 = kernGroupOffset + (((k+2) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex3 = kernGroupOffset + (((k+3) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex4 = kernGroupOffset + (((k+4) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex5 = kernGroupOffset + (((k+5) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex6 = kernGroupOffset + (((k+6) * inChannelGroup + c) * gKernHeight) * gKernWidth;
	    const int64_t kernelIndex7 = kernGroupOffset + (((k+7) * inChannelGroup + c) * gKernHeight) * gKernWidth;

#define INIT_VRSUM_8(TOKEN, INDEX)	\
  __vr vrsum01_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum23_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum45_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;	\
  __vr vrsum67_ ## TOKEN = _vel_vbrdl_vsl(0UL, VLEN) ;

	    INIT_VRSUM_8(r0s0, 0) ;
	    INIT_VRSUM_8(r0s1, 1) ;
	    INIT_VRSUM_8(r0s2, 2) ;
	    INIT_VRSUM_8(r1s0, 3) ;
	    INIT_VRSUM_8(r1s1, 4) ;
	    INIT_VRSUM_8(r1s2, 5) ;
	    INIT_VRSUM_8(r2s0, 6) ;
	    INIT_VRSUM_8(r2s1, 7) ;
	    INIT_VRSUM_8(r2s2, 8) ;
#undef INIT_VRSUM_8

	    for (int64_t n=0; n<batch; n++) {
	      for (int64_t x=0; x<gOutWidth; x+=VLEN) {
		const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;
		const int64_t gOutIndex  = outGroupOffset + ((n * gOutChannel + k  ) * gOutHeight ) * gOutWidth ;

		const int64_t vl = gOutWidth - x < VLEN ? gOutWidth - x : VLEN ;

		__vr vrx = _vel_vaddsl_vsvl(x, _vel_vseq_vl(vl), vl) ;	// op + xy

		__vm256 vmw0_s0 =  _vel_vfmklge_mvl(_vel_vaddsl_vsvl(-1, vrx, vl), vl) ;		// condition(  1<=x)
		__vm256 vmw1_s2 =  _vel_vfmklgt_mvl(_vel_vcmpsl_vsvl(inWidth-1,vrx, vl), vl) ;	// condition(x+1< inWidth)

		__vr vrinP_r1s0, vrinP_r1s1, vrinP_r1s2 ;
		__vr vrinP_r2s0, vrinP_r2s1, vrinP_r2s2 ;

		int64_t y=0 ;
		{
		  const int64_t gop = y * gOutWidth + x ;

		  /* memory access errors mihgt be caused (vrin) */
		  __vr vrin_r1s0    = _vel_vldu_vssl(4,&pInChannel[gop        -1], vl) ;
		  __vr vrin_r1s1    = _vel_vldu_vssl(4,&pInChannel[gop          ], vl) ;
		  __vr vrin_r1s2    = _vel_vldu_vssl(4,&pInChannel[gop        +1], vl) ;
		  __vr vrin_r2s0    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth-1], vl) ;
		  __vr vrin_r2s1    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth  ], vl) ;
		  __vr vrin_r2s2    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth+1], vl) ;

		  __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+0*gOutPixels, vl) ;
		  __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+1*gOutPixels, vl) ;
		  __vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+2*gOutPixels, vl) ;
		  __vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+3*gOutPixels, vl) ;
		  __vr vrgout4 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+4*gOutPixels, vl) ;
		  __vr vrgout5 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+5*gOutPixels, vl) ;
		  __vr vrgout6 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+6*gOutPixels, vl) ;
		  __vr vrgout7 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+7*gOutPixels, vl) ;

		  __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		  __vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
		  __vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
		  __vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

		  vrin_r1s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s0, vmw0_s0, vl) ;
		  vrinP_r1s0 = _vel_vshf_vvvsl(vrin_r1s0, vrin_r1s0, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl) ;
		  vrsum23_r1s0 = _vel_pvfmad_vvvvvl(vrsum23_r1s0, vrinP_r1s0, vrgout23, vrsum23_r1s0, vl) ;
		  vrsum45_r1s0 = _vel_pvfmad_vvvvvl(vrsum45_r1s0, vrinP_r1s0, vrgout45, vrsum45_r1s0, vl) ;
		  vrsum67_r1s0 = _vel_pvfmad_vvvvvl(vrsum67_r1s0, vrinP_r1s0, vrgout67, vrsum67_r1s0, vl) ;

		  // vrin_r1s1 ; // no need to use mask
		  vrinP_r1s1 = _vel_vshf_vvvsl(vrin_r1s1, vrin_r1s1, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl) ;
		  vrsum23_r1s1 = _vel_pvfmad_vvvvvl(vrsum23_r1s1, vrinP_r1s1, vrgout23, vrsum23_r1s1, vl) ;
		  vrsum45_r1s1 = _vel_pvfmad_vvvvvl(vrsum45_r1s1, vrinP_r1s1, vrgout45, vrsum45_r1s1, vl) ;
		  vrsum67_r1s1 = _vel_pvfmad_vvvvvl(vrsum67_r1s1, vrinP_r1s1, vrgout67, vrsum67_r1s1, vl) ;

		  vrin_r1s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r1s2, vmw1_s2, vl) ;
		  vrinP_r1s2 = _vel_vshf_vvvsl(vrin_r1s2, vrin_r1s2, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl) ;
		  vrsum23_r1s2 = _vel_pvfmad_vvvvvl(vrsum23_r1s2, vrinP_r1s2, vrgout23, vrsum23_r1s2, vl) ;
		  vrsum45_r1s2 = _vel_pvfmad_vvvvvl(vrsum45_r1s2, vrinP_r1s2, vrgout45, vrsum45_r1s2, vl) ;
		  vrsum67_r1s2 = _vel_pvfmad_vvvvvl(vrsum67_r1s2, vrinP_r1s2, vrgout67, vrsum67_r1s2, vl) ;

		  vrin_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s0, vmw0_s0, vl) ;
		  vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s0 = _vel_pvfmad_vvvvvl(vrsum01_r2s0, vrinP_r2s0, vrgout01, vrsum01_r2s0, vl) ;
		  vrsum23_r2s0 = _vel_pvfmad_vvvvvl(vrsum23_r2s0, vrinP_r2s0, vrgout23, vrsum23_r2s0, vl) ;
		  vrsum45_r2s0 = _vel_pvfmad_vvvvvl(vrsum45_r2s0, vrinP_r2s0, vrgout45, vrsum45_r2s0, vl) ;
		  vrsum67_r2s0 = _vel_pvfmad_vvvvvl(vrsum67_r2s0, vrinP_r2s0, vrgout67, vrsum67_r2s0, vl) ;

		  // vrin_r2s1 ; // no need to use mask
		  vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s1 = _vel_pvfmad_vvvvvl(vrsum01_r2s1, vrinP_r2s1, vrgout01, vrsum01_r2s1, vl) ;
		  vrsum23_r2s1 = _vel_pvfmad_vvvvvl(vrsum23_r2s1, vrinP_r2s1, vrgout23, vrsum23_r2s1, vl) ;
		  vrsum45_r2s1 = _vel_pvfmad_vvvvvl(vrsum45_r2s1, vrinP_r2s1, vrgout45, vrsum45_r2s1, vl) ;
		  vrsum67_r2s1 = _vel_pvfmad_vvvvvl(vrsum67_r2s1, vrinP_r2s1, vrgout67, vrsum67_r2s1, vl) ;

		  vrin_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s2, vmw1_s2, vl) ;
		  vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP_r2s2, vrgout01, vrsum01_r2s2, vl) ;
		  vrsum23_r2s2 = _vel_pvfmad_vvvvvl(vrsum23_r2s2, vrinP_r2s2, vrgout23, vrsum23_r2s2, vl) ;
		  vrsum45_r2s2 = _vel_pvfmad_vvvvvl(vrsum45_r2s2, vrinP_r2s2, vrgout45, vrsum45_r2s2, vl) ;
		  vrsum67_r2s2 = _vel_pvfmad_vvvvvl(vrsum67_r2s2, vrinP_r2s2, vrgout67, vrsum67_r2s2, vl) ;
		}
		for (y=1; y<gOutHeight-1; y++) {

		  const int64_t gop = y * gOutWidth + x ;

		  /* memory access errors mihgt be caused (vrin) */
		  __vr vrin_r2s0    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth-1], vl) ;
		  __vr vrin_r2s1    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth  ], vl) ;
		  __vr vrin_r2s2    = _vel_vldu_vssl(4,&pInChannel[gop+inWidth+1], vl) ;

		  __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+0*gOutPixels, vl) ;
		  __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+1*gOutPixels, vl) ;
		  __vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+2*gOutPixels, vl) ;
		  __vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+3*gOutPixels, vl) ;
		  __vr vrgout4 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+4*gOutPixels, vl) ;
		  __vr vrgout5 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+5*gOutPixels, vl) ;
		  __vr vrgout6 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+6*gOutPixels, vl) ;
		  __vr vrgout7 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+7*gOutPixels, vl) ;

		  __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		  __vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
		  __vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
		  __vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

		  __vr vrinP_r0s0 = vrinP_r1s0 ;
		  vrsum01_r0s0 = _vel_pvfmad_vvvvvl(vrsum01_r0s0, vrinP_r0s0, vrgout01, vrsum01_r0s0, vl) ;
		  vrsum23_r0s0 = _vel_pvfmad_vvvvvl(vrsum23_r0s0, vrinP_r0s0, vrgout23, vrsum23_r0s0, vl) ;
		  vrsum45_r0s0 = _vel_pvfmad_vvvvvl(vrsum45_r0s0, vrinP_r0s0, vrgout45, vrsum45_r0s0, vl) ;
		  vrsum67_r0s0 = _vel_pvfmad_vvvvvl(vrsum67_r0s0, vrinP_r0s0, vrgout67, vrsum67_r0s0, vl) ;

		  __vr vrinP_r0s1 = vrinP_r1s1 ;
		  vrsum01_r0s1 = _vel_pvfmad_vvvvvl(vrsum01_r0s1, vrinP_r0s1, vrgout01, vrsum01_r0s1, vl) ;
		  vrsum23_r0s1 = _vel_pvfmad_vvvvvl(vrsum23_r0s1, vrinP_r0s1, vrgout23, vrsum23_r0s1, vl) ;
		  vrsum45_r0s1 = _vel_pvfmad_vvvvvl(vrsum45_r0s1, vrinP_r0s1, vrgout45, vrsum45_r0s1, vl) ;
		  vrsum67_r0s1 = _vel_pvfmad_vvvvvl(vrsum67_r0s1, vrinP_r0s1, vrgout67, vrsum67_r0s1, vl) ;

		  __vr vrinP_r0s2 = vrinP_r1s2 ;
		  vrsum01_r0s2 = _vel_pvfmad_vvvvvl(vrsum01_r0s2, vrinP_r0s2, vrgout01, vrsum01_r0s2, vl) ;
		  vrsum23_r0s2 = _vel_pvfmad_vvvvvl(vrsum23_r0s2, vrinP_r0s2, vrgout23, vrsum23_r0s2, vl) ;
		  vrsum45_r0s2 = _vel_pvfmad_vvvvvl(vrsum45_r0s2, vrinP_r0s2, vrgout45, vrsum45_r0s2, vl) ;
		  vrsum67_r0s2 = _vel_pvfmad_vvvvvl(vrsum67_r0s2, vrinP_r0s2, vrgout67, vrsum67_r0s2, vl) ;

		  vrinP_r1s0 = vrinP_r2s0 ;
		  vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl) ;
		  vrsum23_r1s0 = _vel_pvfmad_vvvvvl(vrsum23_r1s0, vrinP_r1s0, vrgout23, vrsum23_r1s0, vl) ;
		  vrsum45_r1s0 = _vel_pvfmad_vvvvvl(vrsum45_r1s0, vrinP_r1s0, vrgout45, vrsum45_r1s0, vl) ;
		  vrsum67_r1s0 = _vel_pvfmad_vvvvvl(vrsum67_r1s0, vrinP_r1s0, vrgout67, vrsum67_r1s0, vl) ;

		  vrinP_r1s1 = vrinP_r2s1 ;
		  vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl) ;
		  vrsum23_r1s1 = _vel_pvfmad_vvvvvl(vrsum23_r1s1, vrinP_r1s1, vrgout23, vrsum23_r1s1, vl) ;
		  vrsum45_r1s1 = _vel_pvfmad_vvvvvl(vrsum45_r1s1, vrinP_r1s1, vrgout45, vrsum45_r1s1, vl) ;
		  vrsum67_r1s1 = _vel_pvfmad_vvvvvl(vrsum67_r1s1, vrinP_r1s1, vrgout67, vrsum67_r1s1, vl) ;

		  vrinP_r1s2 = vrinP_r2s2 ;
		  vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl) ;
		  vrsum23_r1s2 = _vel_pvfmad_vvvvvl(vrsum23_r1s2, vrinP_r1s2, vrgout23, vrsum23_r1s2, vl) ;
		  vrsum45_r1s2 = _vel_pvfmad_vvvvvl(vrsum45_r1s2, vrinP_r1s2, vrgout45, vrsum45_r1s2, vl) ;
		  vrsum67_r1s2 = _vel_pvfmad_vvvvvl(vrsum67_r1s2, vrinP_r1s2, vrgout67, vrsum67_r1s2, vl) ;

		  vrin_r2s0 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s0, vmw0_s0, vl) ;
		  vrinP_r2s0 = _vel_vshf_vvvsl(vrin_r2s0, vrin_r2s0, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s0 = _vel_pvfmad_vvvvvl(vrsum01_r2s0, vrinP_r2s0, vrgout01, vrsum01_r2s0, vl) ;
		  vrsum23_r2s0 = _vel_pvfmad_vvvvvl(vrsum23_r2s0, vrinP_r2s0, vrgout23, vrsum23_r2s0, vl) ;
		  vrsum45_r2s0 = _vel_pvfmad_vvvvvl(vrsum45_r2s0, vrinP_r2s0, vrgout45, vrsum45_r2s0, vl) ;
		  vrsum67_r2s0 = _vel_pvfmad_vvvvvl(vrsum67_r2s0, vrinP_r2s0, vrgout67, vrsum67_r2s0, vl) ;

		  // vrin_r2s1 ; // no need to use mask
		  vrinP_r2s1 = _vel_vshf_vvvsl(vrin_r2s1, vrin_r2s1, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s1 = _vel_pvfmad_vvvvvl(vrsum01_r2s1, vrinP_r2s1, vrgout01, vrsum01_r2s1, vl) ;
		  vrsum23_r2s1 = _vel_pvfmad_vvvvvl(vrsum23_r2s1, vrinP_r2s1, vrgout23, vrsum23_r2s1, vl) ;
		  vrsum45_r2s1 = _vel_pvfmad_vvvvvl(vrsum45_r2s1, vrinP_r2s1, vrgout45, vrsum45_r2s1, vl) ;
		  vrsum67_r2s1 = _vel_pvfmad_vvvvvl(vrsum67_r2s1, vrinP_r2s1, vrgout67, vrsum67_r2s1, vl) ;

		  vrin_r2s2 = _vel_vmrg_vvvml(_vel_vbrds_vsl(0.0f, vl), vrin_r2s2, vmw1_s2, vl) ;
		  vrinP_r2s2 = _vel_vshf_vvvsl(vrin_r2s2, vrin_r2s2, VE_VSHUFFLE_YUZU, vl) ;
		  vrsum01_r2s2 = _vel_pvfmad_vvvvvl(vrsum01_r2s2, vrinP_r2s2, vrgout01, vrsum01_r2s2, vl) ;
		  vrsum23_r2s2 = _vel_pvfmad_vvvvvl(vrsum23_r2s2, vrinP_r2s2, vrgout23, vrsum23_r2s2, vl) ;
		  vrsum45_r2s2 = _vel_pvfmad_vvvvvl(vrsum45_r2s2, vrinP_r2s2, vrgout45, vrsum45_r2s2, vl) ;
		  vrsum67_r2s2 = _vel_pvfmad_vvvvvl(vrsum67_r2s2, vrinP_r2s2, vrgout67, vrsum67_r2s2, vl) ;
		}
		{
		  const int64_t gop = y * gOutWidth + x ;

		  __vr vrgout0 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+0*gOutPixels, vl) ;
		  __vr vrgout1 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+1*gOutPixels, vl) ;
		  __vr vrgout2 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+2*gOutPixels, vl) ;
		  __vr vrgout3 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+3*gOutPixels, vl) ;
		  __vr vrgout4 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+4*gOutPixels, vl) ;
		  __vr vrgout5 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+5*gOutPixels, vl) ;
		  __vr vrgout6 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+6*gOutPixels, vl) ;
		  __vr vrgout7 = _vel_vldu_vssl(4, pGOut+gOutIndex+gop+7*gOutPixels, vl) ;

		  __vr vrgout01 = _vel_vshf_vvvsl(vrgout0, vrgout1, VE_VSHUFFLE_YUZU, vl) ;
		  __vr vrgout23 = _vel_vshf_vvvsl(vrgout2, vrgout3, VE_VSHUFFLE_YUZU, vl) ;
		  __vr vrgout45 = _vel_vshf_vvvsl(vrgout4, vrgout5, VE_VSHUFFLE_YUZU, vl) ;
		  __vr vrgout67 = _vel_vshf_vvvsl(vrgout6, vrgout7, VE_VSHUFFLE_YUZU, vl) ;

		  __vr vrinP_r0s0 = vrinP_r1s0 ;
		  vrsum01_r0s0 = _vel_pvfmad_vvvvvl(vrsum01_r0s0, vrinP_r0s0, vrgout01, vrsum01_r0s0, vl) ;
		  vrsum23_r0s0 = _vel_pvfmad_vvvvvl(vrsum23_r0s0, vrinP_r0s0, vrgout23, vrsum23_r0s0, vl) ;
		  vrsum45_r0s0 = _vel_pvfmad_vvvvvl(vrsum45_r0s0, vrinP_r0s0, vrgout45, vrsum45_r0s0, vl) ;
		  vrsum67_r0s0 = _vel_pvfmad_vvvvvl(vrsum67_r0s0, vrinP_r0s0, vrgout67, vrsum67_r0s0, vl) ;

		  __vr vrinP_r0s1 = vrinP_r1s1 ;
		  vrsum01_r0s1 = _vel_pvfmad_vvvvvl(vrsum01_r0s1, vrinP_r0s1, vrgout01, vrsum01_r0s1, vl) ;
		  vrsum23_r0s1 = _vel_pvfmad_vvvvvl(vrsum23_r0s1, vrinP_r0s1, vrgout23, vrsum23_r0s1, vl) ;
		  vrsum45_r0s1 = _vel_pvfmad_vvvvvl(vrsum45_r0s1, vrinP_r0s1, vrgout45, vrsum45_r0s1, vl) ;
		  vrsum67_r0s1 = _vel_pvfmad_vvvvvl(vrsum67_r0s1, vrinP_r0s1, vrgout67, vrsum67_r0s1, vl) ;

		  __vr vrinP_r0s2 = vrinP_r1s2 ;
		  vrsum01_r0s2 = _vel_pvfmad_vvvvvl(vrsum01_r0s2, vrinP_r0s2, vrgout01, vrsum01_r0s2, vl) ;
		  vrsum23_r0s2 = _vel_pvfmad_vvvvvl(vrsum23_r0s2, vrinP_r0s2, vrgout23, vrsum23_r0s2, vl) ;
		  vrsum45_r0s2 = _vel_pvfmad_vvvvvl(vrsum45_r0s2, vrinP_r0s2, vrgout45, vrsum45_r0s2, vl) ;
		  vrsum67_r0s2 = _vel_pvfmad_vvvvvl(vrsum67_r0s2, vrinP_r0s2, vrgout67, vrsum67_r0s2, vl) ;

		  vrinP_r1s0 = vrinP_r2s0 ;
		  vrsum01_r1s0 = _vel_pvfmad_vvvvvl(vrsum01_r1s0, vrinP_r1s0, vrgout01, vrsum01_r1s0, vl) ;
		  vrsum23_r1s0 = _vel_pvfmad_vvvvvl(vrsum23_r1s0, vrinP_r1s0, vrgout23, vrsum23_r1s0, vl) ;
		  vrsum45_r1s0 = _vel_pvfmad_vvvvvl(vrsum45_r1s0, vrinP_r1s0, vrgout45, vrsum45_r1s0, vl) ;
		  vrsum67_r1s0 = _vel_pvfmad_vvvvvl(vrsum67_r1s0, vrinP_r1s0, vrgout67, vrsum67_r1s0, vl) ;

		  vrinP_r1s1 = vrinP_r2s1 ;
		  vrsum01_r1s1 = _vel_pvfmad_vvvvvl(vrsum01_r1s1, vrinP_r1s1, vrgout01, vrsum01_r1s1, vl) ;
		  vrsum23_r1s1 = _vel_pvfmad_vvvvvl(vrsum23_r1s1, vrinP_r1s1, vrgout23, vrsum23_r1s1, vl) ;
		  vrsum45_r1s1 = _vel_pvfmad_vvvvvl(vrsum45_r1s1, vrinP_r1s1, vrgout45, vrsum45_r1s1, vl) ;
		  vrsum67_r1s1 = _vel_pvfmad_vvvvvl(vrsum67_r1s1, vrinP_r1s1, vrgout67, vrsum67_r1s1, vl) ;

		  vrinP_r1s2 = vrinP_r2s2 ;
		  vrsum01_r1s2 = _vel_pvfmad_vvvvvl(vrsum01_r1s2, vrinP_r1s2, vrgout01, vrsum01_r1s2, vl) ;
		  vrsum23_r1s2 = _vel_pvfmad_vvvvvl(vrsum23_r1s2, vrinP_r1s2, vrgout23, vrsum23_r1s2, vl) ;
		  vrsum45_r1s2 = _vel_pvfmad_vvvvvl(vrsum45_r1s2, vrinP_r1s2, vrgout45, vrsum45_r1s2, vl) ;
		  vrsum67_r1s2 = _vel_pvfmad_vvvvvl(vrsum67_r1s2, vrinP_r1s2, vrgout67, vrsum67_r1s2, vl) ;
		}
	      } // gOutWidth
	    } // batch

	    VSUM_STORE_3X3_UPPER(vrsum01, kernelIndex0) ;
	    VSUM_STORE_3X3_LOWER(vrsum01, kernelIndex1) ;
	    VSUM_STORE_3X3_UPPER(vrsum23, kernelIndex2) ;
	    VSUM_STORE_3X3_LOWER(vrsum23, kernelIndex3) ;
	    VSUM_STORE_3X3_UPPER(vrsum45, kernelIndex4) ;
	    VSUM_STORE_3X3_LOWER(vrsum45, kernelIndex5) ;
	    VSUM_STORE_3X3_UPPER(vrsum67, kernelIndex6) ;
	    VSUM_STORE_3X3_LOWER(vrsum67, kernelIndex7) ;

#undef VSUM_STORE_3X3_UPPER
#undef VSUM_STORE_3X3_LOWER

	  } // inChannel
	} // outChannel
      } // group
    }
  }


  return VEDNN_SUCCESS;
}
