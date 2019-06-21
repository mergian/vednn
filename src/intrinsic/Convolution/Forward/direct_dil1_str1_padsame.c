#include <stdint.h>
#include <stdlib.h>

#include "vednn.h"

#include "veintrin.h"
#define VLEN	(256)

#if 1
vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame(
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
  const int64_t outWidth   = pParamOut->width;		/* must be equal to inWidth */
  const int64_t outHeight  = pParamOut->height;		/* must be equal to inHeight */
  const int64_t kernWidth  = pParamKernel->width;
  const int64_t kernHeight = pParamKernel->height;

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

  const int64_t inChannelGroup  = inChannel  / group;   // equal to pDataKernel->inChannel
  const int64_t outChannelGroup = outChannel / group;   // equal to pDataKernel->outChannel

  const float * restrict pIn     = pDataIn;
  const float * restrict pKernel = pDataKernel;
  float * restrict const pOut    = pDataOut;

  const int oPixels= outHeight*outWidth ;


  {
    for (int64_t n = 0; n < batch; n++) {
      for (int64_t g = 0; g < group; g++) {
	const int64_t inGroupOffset   = g * inChannelGroup * inHeight * inWidth;
	const int64_t outGroupOffset  = g * outChannelGroup * outHeight * outWidth;
	const int64_t kernGroupOffset = g * outChannelGroup * inChannelGroup * kernHeight * kernWidth;

	int k = 0 ;
	if ( (outChannelGroup & 0x01) == 1 ) {
	  int64_t outIndex = outGroupOffset + (n * outChannel + k) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vrsum = _ve_vbrdu_vs_f32(0.0f) ;
	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _ve_vaddsl_vsv(r-padHeight, vry) ;
		__vr vrw = _ve_vaddsl_vsv(s-padWidth,  vrx) ;

		__vm256 vm0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		__vm256 vm1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		__vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		__vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		__vm256 vm01  = _ve_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _ve_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		  /* memory access errors mihgt be caused */
		  __vr vrin = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth+s-padWidth]) ;
		  vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrin) ;

	        } // kernWidth
	      } // kernHeight
	    } // inChannel

	    _ve_vstu_vss(vrsum, 4, pOut+outIndex) ;

	    outIndex += vl ;
	  } // outPixels
	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _ve_vaddsl_vsv(r-padHeight, vry) ;
		__vr vrw = _ve_vaddsl_vsv(s-padWidth,  vrx) ;

		__vm256 vm0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		__vm256 vm1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		__vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		__vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		__vm256 vm01  = _ve_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _ve_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		  /* memory access errors mihgt be caused */
		  __vr vrin = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth+s-padWidth]) ;
		  vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		  __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
							    pKerValue+      inChannelGroup * kernHeight * kernWidth) ;

		  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
		} // inChannel
	      } // kernWidth
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;

	    outIndex0 += vl ;
	    outIndex1 += vl ;
	  } // outPixels

	  k+=2 ;
	}
	if ( ((outChannelGroup >> 2) & 0x01) == 1 ) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
	  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
	  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _ve_vaddsl_vsv(r-padHeight, vry) ;
		__vr vrw = _ve_vaddsl_vsv(s-padWidth,  vrx) ;

		__vm256 vm0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		__vm256 vm1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		__vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		__vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		__vm256 vm01  = _ve_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _ve_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		  /* memory access errors mihgt be caused */
		  __vr vrin = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth+s-padWidth]) ;
		  vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		  __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
							    pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;

		  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
		  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrinP) ;
		} // inChannel

	      } // kernWidth
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex2) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex3) ;

	    outIndex0 += vl ;
	    outIndex1 += vl ;
	    outIndex2 += vl ;
	    outIndex3 += vl ;
	  } // outPixels

	  k+=4 ;
	}
	for ( ; k < outChannelGroup; k+=8) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * oPixels ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * oPixels ;
	  int64_t outIndex2 = outGroupOffset + (n * outChannel + k+2) * oPixels ;
	  int64_t outIndex3 = outGroupOffset + (n * outChannel + k+3) * oPixels ;
	  int64_t outIndex4 = outGroupOffset + (n * outChannel + k+4) * oPixels ;
	  int64_t outIndex5 = outGroupOffset + (n * outChannel + k+5) * oPixels ;
	  int64_t outIndex6 = outGroupOffset + (n * outChannel + k+6) * oPixels ;
	  int64_t outIndex7 = outGroupOffset + (n * outChannel + k+7) * oPixels ;

	  for (int64_t op = 0; op < oPixels; op+=VLEN) {
	    const int64_t vl = oPixels - op < VLEN ? oPixels - op : VLEN ;

	    _ve_lvl(vl) ;

	    __vr vrseq = _ve_vseq_v() ;			// xy
	    __vr vridx = _ve_vaddsl_vsv(op, vrseq) ;	// op + xy

	    __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	    __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

	    __vr vry   = _ve_vdivsl_vvs(vridx, outWidth) ;
	    __vr vrx   = _ve_vsubsl_vvv(vridx, _ve_vmulul_vsv(outWidth,vry)) ;

	    for (int64_t r = 0; r < kernHeight; r++) {
	      for (int64_t s = 0; s < kernWidth; s++) {
		__vr vrh = _ve_vaddsl_vsv(r-padHeight, vry) ;
		__vr vrw = _ve_vaddsl_vsv(s-padWidth,  vrx) ;

		__vm256 vm0 = _ve_vfmkl_mcv(VECC_GE, vrh) ;				// condition(0 <= h)
		__vm256 vm1 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inHeight,vrh)) ;	// condition(h < inHeight)
		__vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		__vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		__vm256 vm01  = _ve_andm_mmm(vm0, vm1) ;
		__vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;
		__vm256 vmall = _ve_andm_mmm(vm01,vm23) ;

		for (int64_t c = 0; c < inChannelGroup; c++) {
		  const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		  /* memory access errors mihgt be caused */
		  __vr vrin = _ve_vldu_vss(4,&pInChannel[op+(r-padHeight)*inWidth+s-padWidth]) ;
		  vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vmall) ;

		  __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		  const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		  const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
							    pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 5 * inChannelGroup * kernHeight * kernWidth) ;
		  const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth,
							    pKerValue + 7 * inChannelGroup * kernHeight * kernWidth) ;

		  vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
		  vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrinP) ;
		  vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrinP) ;
		  vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrinP) ;
		} // inChannel
	      } // kernWidth
	    } // kernHeight

	    _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
	    _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;
	    _ve_vstu_vss(vrsum23, 4, pOut+outIndex2) ;
	    _ve_vstl_vss(vrsum23, 4, pOut+outIndex3) ;
	    _ve_vstu_vss(vrsum45, 4, pOut+outIndex4) ;
	    _ve_vstl_vss(vrsum45, 4, pOut+outIndex5) ;
	    _ve_vstu_vss(vrsum67, 4, pOut+outIndex6) ;
	    _ve_vstl_vss(vrsum67, 4, pOut+outIndex7) ;

	    outIndex0 += vl ;
	    outIndex1 += vl ;
	    outIndex2 += vl ;
	    outIndex3 += vl ;
	    outIndex4 += vl ;
	    outIndex5 += vl ;
	    outIndex6 += vl ;
	    outIndex7 += vl ;
	  } // outPixels
	} // outChannel
      } // group
    } // batch
  }

  return VEDNN_SUCCESS;
}

#else
vednnError_t
vednnConvolutionForward_direct_dil1_str1_padsame(
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
  const int64_t kernWidth  = pParamKernel->width;		/* must be 2*padWidth  + 1 */
  const int64_t kernHeight = pParamKernel->height;		/* must be 2*padHeight + 1 */

  const int64_t group          = pParamConv->group;
//  const int64_t strideWidth    = pParamConv->strideWidth;	/* must be 1 */
//  const int64_t strideHeight   = pParamConv->strideHeight;	/* must be 1 */
  const int64_t padWidth       = pParamConv->padWidth;
  const int64_t padHeight      = pParamConv->padHeight;
//  const int64_t dilationWidth  = pParamConv->dilationWidth;	/* must be 1 */
//  const int64_t dilationHeight = pParamConv->dilationHeight;	/* must be 1 */

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
	    int64_t i = y - padHeight;

	    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

	      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

	      _ve_lvl(vl) ;

	      __vr vrseq = _ve_vseq_v() ;
	      __vr vrsum = _ve_vbrdu_vs_f32(0.0f) ;
	      __vr vrj   = _ve_vaddsl_vsv(-padWidth, _ve_vaddsl_vsv(x0, vrseq)) ;

	      for (int64_t r = 0; r < kernHeight; r++) {
		for (int64_t s = 0; s < kernWidth; s++) {

		  const int64_t h = i + r ;
		  if ( !(h < 0 || inHeight <= h) ) {
		    __vr vrw = _ve_vaddsl_vsv(s,  vrj) ;

		    __vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		    __vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		    __vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;

		    for (int64_t c = 0; c < inChannelGroup; c++) {
		      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		      /* memory access errors mihgt be caused */
		      __vr vrin = _ve_vldu_vss(4,&pInChannel[h*inWidth+x0+s-padWidth]) ;
		      vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vm23) ;

		      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		      vrsum = _ve_vfmads_vvsv(vrsum, *pKerValue, vrin) ;
		    } // inChannel
		  }
		} // kernWidth
	      } // kernHeight


	      _ve_vstu_vss(vrsum, 4, pOut+outIndex) ;

	      outIndex += vl ;
	    } // x
	  } // y

	  k++ ;
	}
	if ( ((outChannelGroup >> 1) & 0x01) == 1 ) {
	  int64_t outIndex0 = outGroupOffset + (n * outChannel + k  ) * outHeight*outWidth ;
	  int64_t outIndex1 = outGroupOffset + (n * outChannel + k+1) * outHeight*outWidth ;

	  for (int64_t y=0; y<outHeight; y++) {
	    int64_t i = y - padHeight;

	    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

	      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

	      _ve_lvl(vl) ;

	      __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;

	      __vr vrseq = _ve_vseq_v() ;
	      __vr vrj   = _ve_vaddsl_vsv(-padWidth,  _ve_vaddsl_vsv(x0, vrseq)) ;

	      for (int64_t r = 0; r < kernHeight; r++) {
		for (int64_t s = 0; s < kernWidth; s++) {

		  const int64_t h = i + r ;
		  if ( !(h < 0 || inHeight <= h) ) {
		    __vr vrw = _ve_vaddsl_vsv(s,  vrj) ;

		    __vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		    __vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		    __vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;

		    for (int64_t c = 0; c < inChannelGroup; c++) {
		      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		      /* memory access errors mihgt be caused */
		      __vr vrin = _ve_vldu_vss(4,&pInChannel[h*inWidth+x0+s-padWidth]) ;
		      vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vm23) ;
		      __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		      const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
			                                        pKerValue+      inChannelGroup * kernHeight * kernWidth) ;

		      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
		    } // inChannel
		  }
		} // kernWidth
	      } // kernHeight

	      _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
	      _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;

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
	    int64_t i = y - padHeight;

	    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

	      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

	      _ve_lvl(vl) ;

	      __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	      __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;

	      __vr vrseq = _ve_vseq_v() ;
	      __vr vrj   = _ve_vaddsl_vsv(-padWidth,  _ve_vaddsl_vsv(x0, vrseq)) ;

	      for (int64_t r = 0; r < kernHeight; r++) {
		for (int64_t s = 0; s < kernWidth; s++) {
		  const int64_t h = i + r ;
		  if ( !(h < 0 || inHeight <= h) ) {
		    __vr vrw = _ve_vaddsl_vsv(s,  vrj) ;

		    __vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		    __vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		    __vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;

		    for (int64_t c = 0; c < inChannelGroup; c++) {
		      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		      /* memory access errors mihgt be caused */
		      __vr vrin = _ve_vldu_vss(4,&pInChannel[h*inWidth+x0+s-padWidth]) ;
		      vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vm23) ;
		      __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		      const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
			                                        pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
		      const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
			                                        pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;

		      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
		      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrinP) ;
		    } // inChannel
		  }
		} // kernWidth
	      } // kernHeight

	      _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
	      _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;
	      _ve_vstu_vss(vrsum23, 4, pOut+outIndex2) ;
	      _ve_vstl_vss(vrsum23, 4, pOut+outIndex3) ;

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
	    int64_t i = y - padHeight;

	    for ( int64_t x0=0; x0<outWidth; x0+=VLEN) {

	      const int64_t vl = outWidth - x0 < VLEN ? outWidth - x0 : VLEN ;

	      _ve_lvl(vl) ;

	      __vr vrsum01 = _ve_pvbrd_vs_i64(0UL) ;
	      __vr vrsum23 = _ve_pvbrd_vs_i64(0UL) ;
	      __vr vrsum45 = _ve_pvbrd_vs_i64(0UL) ;
	      __vr vrsum67 = _ve_pvbrd_vs_i64(0UL) ;

	      __vr vrseq = _ve_vseq_v() ;
	      __vr vrj   = _ve_vaddsl_vsv(-padWidth, _ve_vaddsl_vsv(x0, vrseq)) ;

	      const float *pKerValue = pKernel + kernGroupOffset + k * inChannelGroup * kernHeight * kernWidth ;

	      for (int64_t r = 0; r < kernHeight; r++) {
		for (int64_t s = 0; s < kernWidth; s++) {

		  const int64_t h = i + r ;
		  if ( !(h < 0 || inHeight <= h) ) {
		    __vr vrw = _ve_vaddsl_vsv(s,  vrj) ;

		    __vm256 vm2 = _ve_vfmkl_mcv(VECC_GE, vrw) ;				// condition(0 <= w)
		    __vm256 vm3 = _ve_vfmkl_mcv(VECC_IG, _ve_vcmpsl_vsv(inWidth,vrw)) ;	// condition(w < inWidth)

		    __vm256 vm23  = _ve_andm_mmm(vm2, vm3) ;

		    for (int64_t c = 0; c < inChannelGroup; c++) {
		      const float *pInChannel = pIn + inGroupOffset + ((n * inChannel + c) * inHeight * inWidth ) ;

		      /* memory access errors mihgt be caused */
		      __vr vrin = _ve_vldu_vss(4,&pInChannel[h*inWidth+x0+s-padWidth]) ;
		      vrin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.0f), vrin, vm23) ;
		      __vr vrinP = _ve_vshf_vvvs(vrin, vrin, VE_VSHUFFLE_YUZU) ;

		      const float *pKerValue = pKernel + kernGroupOffset + ((k * inChannelGroup + c) * kernHeight + r) * kernWidth + s;

		      const uint64_t kerValue01 = _ve_pack_f32p(pKerValue,
			                                        pKerValue+      inChannelGroup * kernHeight * kernWidth) ;
		      const uint64_t kerValue23 = _ve_pack_f32p(pKerValue + 2 * inChannelGroup * kernHeight * kernWidth,
			                                        pKerValue + 3 * inChannelGroup * kernHeight * kernWidth) ;
		      const uint64_t kerValue45 = _ve_pack_f32p(pKerValue + 4 * inChannelGroup * kernHeight * kernWidth,
			                                        pKerValue + 5 * inChannelGroup * kernHeight * kernWidth) ;
		      const uint64_t kerValue67 = _ve_pack_f32p(pKerValue + 6 * inChannelGroup * kernHeight * kernWidth,
			                                        pKerValue + 7 * inChannelGroup * kernHeight * kernWidth) ;

		      vrsum01 = _ve_pvfmad_vvsv(vrsum01, kerValue01, vrinP) ;
		      vrsum23 = _ve_pvfmad_vvsv(vrsum23, kerValue23, vrinP) ;
		      vrsum45 = _ve_pvfmad_vvsv(vrsum45, kerValue45, vrinP) ;
		      vrsum67 = _ve_pvfmad_vvsv(vrsum67, kerValue67, vrinP) ;
		    } // inChannel
		  }
		} // kernWidth
	      } // kernHeight

	      _ve_vstu_vss(vrsum01, 4, pOut+outIndex0) ;
	      _ve_vstl_vss(vrsum01, 4, pOut+outIndex1) ;
	      _ve_vstu_vss(vrsum23, 4, pOut+outIndex2) ;
	      _ve_vstl_vss(vrsum23, 4, pOut+outIndex3) ;
	      _ve_vstu_vss(vrsum45, 4, pOut+outIndex4) ;
	      _ve_vstl_vss(vrsum45, 4, pOut+outIndex5) ;
	      _ve_vstu_vss(vrsum67, 4, pOut+outIndex6) ;
	      _ve_vstl_vss(vrsum67, 4, pOut+outIndex7) ;

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
#endif
