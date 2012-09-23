#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialSAD.c"
#else
#if defined USE_SSE2 || defined USE_SSE3 || defined USE_SSE3 \
  || defined USE_SSE4_1 || defined USE_SSE4_2
#include <emmintrin.h> //SSE2
//#include <immintrin.h> // AVX
#ifndef MM_ABS
#define MM_ABS
inline __m128d _mm_abs_pd(__m128d m) {
	return _mm_andnot_pd(_mm_set1_pd(-0.0), m);
}
inline __m128 _mm_abs_ps(__m128 m) {
	return _mm_andnot_ps(_mm_set1_ps(-0.0f), m);
}
//static inline __m128d _mm_abs_pd( register __m128d a )
//{ const static long long am1[2] = {~0x8000000000000000LL,~0x8000000000000000LL};
//  return _mm_and_pd(a, *((__m128d*)am1) );
//}
//static inline __m256d _mm256_abs_pd( register __m256d a )
//{ const static long long am1[4] = {~0x8000000000000000LL,~0x8000000000000000LL,~0x8000000000000000LL,~0x8000000000000000LL};
//  return _mm256_and_pd(a, *((__m256d*)am1) );
//}
#endif // mm_abs
#endif // sse
static int nn_(SpatialSAD_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  //THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  input = THTensor_(newContiguous)(input);
  weight = THTensor_(newContiguous)(weight);

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");

  int dimh = 1;
  int dimw = 2;
  if (input->nDimension == 4) {
    dimh++;
    dimw++;
  }

  long nOutputPlane = weight->size[0];
  long nKernelRows = weight->size[2];
  long nKernelCols = weight->size[3];
  long kstride0    = weight->stride[0];
  long kstride1    = weight->stride[1];

  long nInputPlane = input->size[0];
  long nInputRows  = input->size[dimh];
  long nInputCols   = input->size[dimw];
  long istride0    = input->stride[0];
  long istride1    = input->stride[1];

  long nOutputRows = (nInputRows - nKernelRows) / dH + 1;
  long nOutputCols  = (nInputCols - nKernelCols) / dW + 1;

  THArgCheck(weight->size[1] == nInputPlane, 2, "invalid number of input planes");
  // TODO: check that # kernel channels is correct
  THArgCheck( (nInputRows >= nKernelRows && nInputCols >= nKernelCols) , 2, "SpatialSAD : Input image is smaller than kernel");

  if (input->nDimension == 3)
  {
    THTensor_(resize3d)(output, nOutputPlane, nOutputRows, nOutputCols);

    real *input_data = THTensor_(data)(input);
    real *weight_data = THTensor_(data)(weight);
    real *output_data = THTensor_(data)(output);

    // fill output with zero values
    long k;
#pragma omp parallel for private(k)
    for (k = 0; k < nOutputPlane; k++)
    {
      real *ptr_output = output_data + k*nOutputCols*nOutputRows;
      long l;
      for (l = 0; l < nOutputRows*nOutputCols; l++)
        ptr_output[l] = 0.0;
    }

#if defined USE_SSE2 || defined USE_SSE3 || defined USE_SSE3 || defined USE_SSE4_1 || defined USE_SSE4_2
    if ((dW != 1) || (nOutputCols < 4)) {
#endif
      /* regular loop */
      long zz,yy,xx,ky,kx;
#pragma omp parallel for private(k,zz,yy,xx,ky,kx)
      for(k = 0; k < nOutputPlane; k++)
      {
        for(zz = 0; zz < nInputPlane; zz++)
        {
          real *ptr_output = output_data + k*nOutputRows*nOutputCols;
          for(yy = 0; yy < nOutputRows; yy++) {
            for(xx = 0; xx < nOutputCols; xx++) {
              real *ptr_input = input_data + zz*istride0 + yy*dH*istride1 + dW*xx;
              //printf("input = %f\n",*ptr_input);
              real *ptr_weight = weight_data + k*kstride0 + zz*kstride1;
              //printf("weight = %f\n",*ptr_weight);
              accreal sum = 0;
              for(ky = 0; ky < nKernelRows; ky++) {
                for(kx = 0; kx < nKernelCols; kx++) {
                  sum += fabs(ptr_input[kx]-ptr_weight[kx]);
                }
                ptr_input += nInputCols; /* next input line */
                ptr_weight += nKernelCols; /* next mask line */
              }
              *ptr_output++ += sum;
              //printf("sum = %f\n",sum);
            }
          }
        /* Next output plane */
        /* output_data += nOutputCols*nOutputRows;*/
        }
      }
#if defined USE_SSE2 || defined USE_SSE3 || defined USE_SSE3 || defined USE_SSE4_1 || defined USE_SSE4_2
    }
    else { // sse loops
      long zz,yy,kx,ky,j;
#pragma omp parallel for private(k,zz,yy,ky,kx,j)
      for(k = 0; k < nOutputPlane; k++)
      {
        for(zz = 0; zz < nInputPlane; zz++)
        {
  //#pragma omp parallel for private(yy,ky,kx,j)
          for(yy = 0; yy < nOutputRows; y++) {
            real *ptr_output = output_data + k*nOutputRows*nOutputCols + yy*nOutputCols;
            real *ptr_weight = weight_data + k*kstride0 + zz*kstride1;
            real *ptr_input = input_data + zz*istride0 + yy*dH*istride1;
            for(ky = 0; ky < nKernelRows; ky++) {
              for(kx = 0; kx < nKernelCols; kx++) {
                real *ptr_is = ptr_input + kx;
                real *ptr_os = ptr_output;
                /* regular code */
                //for(j = 0; j < nOutputCols; j++) {
                //  //real ad = abs(*ptr_is-ptr_weight[kx]);
                //  //printf("*ptr_os = %f, adding abs(*ptr_is = %f - ptr_weight[kx] = %f) = %f ", *ptr_os, *ptr_is, ptr_weight[kx], ad); 
                //  //printf("k = %ld,zz = %ld,yy = %ld,ky = %ld,kx = %ld,j = %ld\n",k,zz,yy,ky,kx,j);
                //  *ptr_os += abs(*ptr_is-ptr_weight[kx]);
                //  ptr_is++;
                //  ptr_os++;
                //}
                /* SSE instructions */
#if defined(TH_REAL_IS_DOUBLE)
                __m128d XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8,XMM9,XMM10,XMM11,XMM12,XMM13;
                __m128d XMM15 = _mm_set1_pd(ptr_weight[kx]);
                const long blocksz = 4;
                for(j = 0; j < (nOutputCols-blocksz); j+=blocksz) {
                  XMM0 = _mm_loadu_pd(ptr_is  );
                  XMM0 = _mm_abs_pd(_mm_sub_pd(XMM0, XMM15));
                  XMM1 = _mm_loadu_pd(ptr_os  );
                  XMM1 = _mm_add_pd(XMM1, XMM0);
                  _mm_storeu_pd(ptr_os,   XMM1);
                  XMM2 = _mm_loadu_pd(ptr_is+2);
                  XMM2 = _mm_abs_pd(_mm_sub_pd(XMM2, XMM15));
                  XMM3 = _mm_loadu_pd(ptr_os+2);
                  XMM3 = _mm_add_pd(XMM3, XMM2);
                  _mm_storeu_pd(ptr_os+2, XMM3);
                  //XMM4 = _mm_loadu_pd(ptr_is+4);
                  //XMM4 = _mm_abs_pd(_mm_sub_pd(XMM4, XMM15));
                  //XMM5 = _mm_loadu_pd(ptr_os+4);
                  //XMM5 = _mm_add_pd(XMM5, XMM4);
                  //_mm_storeu_pd(ptr_os+4, XMM5);
                  //XMM6 = _mm_loadu_pd(ptr_is+6);
                  //XMM6 = _mm_abs_pd(_mm_sub_pd(XMM6, XMM15));
                  //XMM7 = _mm_loadu_pd(ptr_os+6);
                  //XMM7 = _mm_add_pd(XMM7, XMM6);
                  //_mm_storeu_pd(ptr_os+6, XMM7);
                  //XMM8 = _mm_loadu_pd(ptr_is+8);
                  //XMM8 = _mm_abs_pd(_mm_sub_pd(XMM8, XMM15));
                  //XMM9 = _mm_loadu_pd(ptr_os+8);
                  //XMM9 = _mm_add_pd(XMM9, XMM8);
                  //_mm_storeu_pd(ptr_os+8, XMM9);
                  //XMM10 = _mm_loadu_pd(ptr_is+10);
                  //XMM10 = _mm_abs_pd(_mm_sub_pd(XMM10, XMM15));
                  //XMM11 = _mm_loadu_pd(ptr_os+10);
                  //XMM11 = _mm_add_pd(XMM11, XMM10);
                  //_mm_storeu_pd(ptr_os+10,XMM11);
                  //XMM12 = _mm_loadu_pd(ptr_is+12);
                  //XMM12 = _mm_abs_pd(_mm_sub_pd(XMM12, XMM15));
                  //XMM13 = _mm_loadu_pd(ptr_os+12);
                  //XMM13 = _mm_add_pd(XMM13, XMM12);
                  //_mm_storeu_pd(ptr_os+12,XMM13);
                  ptr_is+=blocksz;
                  ptr_os+=blocksz;
                }
#elif defined(TH_REAL_IS_FLOAT)
                __m128 XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8,XMM9,XMM10,XMM11,XMM12,XMM13;
                __m128 XMM15 = _mm_set1_ps(ptr_weight[kx]);
                const long blocksz = 8;
                for(j = 0; j < (nOutputCols-blocksz); j+=blocksz) {
                  XMM0 = _mm_loadu_ps(ptr_is  );
                  XMM0 = _mm_abs_ps(_mm_sub_ps(XMM0, XMM15));
                  XMM1 = _mm_loadu_ps(ptr_os  );
                  XMM1 = _mm_add_ps(XMM1, XMM0);
                  _mm_storeu_ps(ptr_os,   XMM1);
                  XMM2 = _mm_loadu_ps(ptr_is+4);
                  XMM2 = _mm_abs_ps(_mm_sub_ps(XMM2, XMM15));
                  XMM3 = _mm_loadu_ps(ptr_os+4);
                  XMM3 = _mm_add_ps(XMM3, XMM2);
                  _mm_storeu_ps(ptr_os+4, XMM3);
                  //XMM4 = _mm_loadu_ps(ptr_is+8);
                  //XMM4 = _mm_abs_ps(_mm_sub_ps(XMM4, XMM15));
                  //XMM5 = _mm_loadu_ps(ptr_os+8);
                  //XMM5 = _mm_add_ps(XMM5, XMM4);
                  //_mm_storeu_ps(ptr_os+8, XMM5);
                  //XMM6 = _mm_loadu_ps(ptr_is+12);
                  //XMM6 = _mm_abs_ps(_mm_sub_ps(XMM6, XMM15));
                  //XMM7 = _mm_loadu_ps(ptr_os+12);
                  //XMM7 = _mm_add_ps(XMM7, XMM6);
                  //_mm_storeu_ps(ptr_os+12, XMM7);
                  //XMM8 = _mm_loadu_ps(ptr_is+16);
                  //XMM8 = _mm_abs_ps(_mm_sub_ps(XMM8, XMM15));
                  //XMM9 = _mm_loadu_ps(ptr_os+16);
                  //XMM9 = _mm_add_ps(XMM9, XMM8);
                  //_mm_storeu_ps(ptr_os+16, XMM9);
                  //XMM10 = _mm_loadu_ps(ptr_is+20);
                  //XMM10 = _mm_abs_ps(_mm_sub_ps(XMM10, XMM15));
                  //XMM11 = _mm_loadu_ps(ptr_os+20);
                  //XMM11 = _mm_add_ps(XMM11, XMM10);
                  //_mm_storeu_ps(ptr_os+20,XMM11);
                  //XMM12 = _mm_loadu_ps(ptr_is+24);
                  //XMM12 = _mm_abs_ps(_mm_sub_ps(XMM12, XMM15));
                  //XMM13 = _mm_loadu_ps(ptr_os+24);
                  //XMM13 = _mm_add_ps(XMM13, XMM12);
                  //_mm_storeu_ps(ptr_os+24,XMM13);
                  ptr_is+=blocksz;
                  ptr_os+=blocksz;
                }
#endif // end type specific code
                /* AVX instructions */
                //__m256d XMM7 = _mm256_set1_pd(ptr_weight[kx]);
                //__m256d XMM0, XMM1, XMM2, XMM3, XMM4, XMM5;
                //const long blocksz = 4;
                //for(j = 0; j < (nOutputCols-blocksz); j+=blocksz) {
                //  //printf("k = %ld,zz = %ld,yy = %ld,ky = %ld,kx = %ld,j = %ld\n",k,zz,yy,ky,kx,j);
                //  XMM0 = _mm256_loadu_pd(ptr_is  );
                //  XMM0 = _mm256_abs_pd(_mm256_sub_pd(XMM0, XMM7));
                //  XMM3 = _mm256_loadu_pd(ptr_os  );
                //  XMM3 = _mm256_add_pd(XMM3, XMM0);
                //  _mm256_storeu_pd(ptr_os,   XMM3);
                //  //XMM1 = _mm256_loadu_pd(ptr_is+4);
                //  //XMM1 = _mm256_abs_pd(_mm256_sub_pd(XMM1, XMM7));
                //  //XMM4 = _mm256_loadu_pd(ptr_os+4);
                //  //XMM4 = _mm256_add_pd(XMM4, XMM1);
                //  //_mm256_storeu_pd(ptr_os,   XMM4);
                //  ptr_is+=blocksz;
                //  ptr_os+=blocksz;
                //}
                for (; j<nOutputCols; j++) {
                  *ptr_os += fabs(*ptr_is-ptr_weight[kx]);
                  ptr_is++;
                  ptr_os++;
                }
              }
              ptr_weight += nKernelCols; /* next mask line */
              ptr_input += nInputCols; /* next input line */
            }
          }
        } 
      }
    } // end sse loops
#endif // sse loops
  }
  else
  {
    // TODO: rewrite batch mode
    //THTensor_(resize4d)(output, input->size[0], nOutputPlane, nOutputRows, nOutputCols);

  }

  THTensor_(free)(input);
  THTensor_(free)(weight);
  return 1;
}


static int nn_(SpatialSAD_updateGradInput)(lua_State *L)
{
  // TODO: Rewrite
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  return 1;
}


static int nn_(SpatialSAD_accGradParameters)(lua_State *L)
{
  // TODO: Rewrite
  return 0;
}

static const struct luaL_Reg nn_(SpatialSAD__) [] = {
  {"SpatialSAD_updateOutput", nn_(SpatialSAD_updateOutput)},
  {"SpatialSAD_updateGradInput", nn_(SpatialSAD_updateGradInput)},
  {"SpatialSAD_accGradParameters", nn_(SpatialSAD_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SpatialSAD_init)(lua_State *L)
{
  // TODO: Rewrite?
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialSAD__), "nn");
  lua_pop(L,1);
}

#endif
