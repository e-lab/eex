#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/sad.c"
#else

#if defined USE_SSE2 || defined USE_SSE3 || defined USE_SSSE3 \
  || defined USE_SSE4_1 || defined USE_SSE4_2

#ifdef USE_SSE2
#include <emmintrin.h>
#endif

#ifdef USE_SSE3
#include <pmmintrin.h>
#endif

#ifdef USE_SSSE3
#include <tmmintrin.h>
#endif

#if defined (USE_SSE4_2) || defined (USE_SSE4_1)
#include <smmintrin.h>
#endif

#ifndef MM_ABS
#define MM_ABS
static inline __m128d _mm_abs_pd(__m128d m) {
	return _mm_andnot_pd(_mm_set1_pd(-0.0), m);
}
static inline __m128 _mm_abs_ps(__m128 m) {
	return _mm_andnot_ps(_mm_set1_ps(-0.0f), m);
}
#endif // mm_abs

#endif // sse defs



void eex_(SAD)(real *output_data, const long output_h, const long output_w, 
               const real *input_data, const long input_w,
               const real *weight_data, const long weight_h, const long weight_w,
               const int dH, const int dW)
{
#if defined USE_SSE2 || defined USE_SSE3 || defined USE_SSSE3 || defined USE_SSE4_1 || defined USE_SSE4_2
  if((dW != 1) || (output_w < 4)) { // TODO: write sse vectorized SAD for dW > 1
#endif
    real *ptr_output = output_data;
    long yy,xx,ky,kx;
    for(yy = 0; yy < output_h; yy++) {
      for(xx = 0; xx < output_w; xx++) {
        const real *ptr_input = input_data + yy*dH*input_w + dW*xx;
        const real *ptr_weight = weight_data;
        accreal sum = 0;
        for(ky = 0; ky < weight_h; ky++) {
          for(kx = 0; kx < weight_w; kx++) {
            sum += fabs(ptr_input[kx]-ptr_weight[kx]);
          }
          ptr_input += input_w; /* next input line */
          ptr_weight += weight_w; /* next mask line */
        }
        *ptr_output++ += sum;
      }
    }
#if defined USE_SSE2 || defined USE_SSE3 || defined USE_SSSE3 || defined USE_SSE4_1 || defined USE_SSE4_2
  }
  else {
    long yy,ky,kx,j;
    for(yy = 0; yy < output_h; yy++) {
      real *ptr_output = output_data + yy*output_w;
      const real *ptr_input = input_data + yy*dH*input_w;
      const real *ptr_weight = weight_data;
      for(ky = 0; ky < weight_h; ky++) {
        for(kx = 0; kx < weight_w; kx++) {
          const real *ptr_is = ptr_input + kx;
          real *ptr_os = ptr_output;
#if defined(TH_REAL_IS_DOUBLE)
          __m128d XMM0, XMM1, XMM2, XMM3;
          __m128d XMM4 = _mm_set1_pd(ptr_weight[kx]);
          const long blocksz = 4;
          for(j = 0; j < (output_w-blocksz); j+=blocksz) {
            XMM0 = _mm_loadu_pd(ptr_is  );
            XMM0 = _mm_abs_pd(_mm_sub_pd(XMM0, XMM4));
            XMM1 = _mm_loadu_pd(ptr_os  );
            XMM1 = _mm_add_pd(XMM1, XMM0);
            _mm_storeu_pd(ptr_os,   XMM1);
            XMM2 = _mm_loadu_pd(ptr_is+2);
            XMM2 = _mm_abs_pd(_mm_sub_pd(XMM2, XMM4));
            XMM3 = _mm_loadu_pd(ptr_os+2);
            XMM3 = _mm_add_pd(XMM3, XMM2);
            _mm_storeu_pd(ptr_os+2, XMM3);
            ptr_is+=blocksz;
            ptr_os+=blocksz;
          }
#elif defined(TH_REAL_IS_FLOAT)
          __m128 XMM0, XMM1, XMM2, XMM3;
          __m128 XMM4 = _mm_set1_ps(ptr_weight[kx]);
          const long blocksz = 8;
          for(j = 0; j < (output_w-blocksz); j+=blocksz) {
            XMM0 = _mm_loadu_ps(ptr_is  );
            XMM0 = _mm_abs_ps(_mm_sub_ps(XMM0, XMM4));
            XMM1 = _mm_loadu_ps(ptr_os  );
            XMM1 = _mm_add_ps(XMM1, XMM0);
            _mm_storeu_ps(ptr_os,   XMM1);
            XMM2 = _mm_loadu_ps(ptr_is+4);
            XMM2 = _mm_abs_ps(_mm_sub_ps(XMM2, XMM4));
            XMM3 = _mm_loadu_ps(ptr_os+4);
            XMM3 = _mm_add_ps(XMM3, XMM2);
            _mm_storeu_ps(ptr_os+4, XMM3);
            ptr_is+=blocksz;
            ptr_os+=blocksz;
          }
#endif // end type specific code
          for (; j<output_w; j++) {
            *ptr_os += fabs(*ptr_is-ptr_weight[kx]);
            ptr_is++;
            ptr_os++;
          }
        }
        ptr_input += input_w; /* next input line */
        ptr_weight += weight_w; /* next mask line */
      }
    }
  }
#endif //SSE
}

#endif
