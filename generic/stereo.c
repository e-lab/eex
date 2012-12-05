#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define abs(a)    (a) < 0 ? -(a) : (a)

static int l_stereo(lua_State *L)
{

  // get args
  const void* torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
  THFloatTensor *dispMapX_ptr = luaT_checkudata(L, 1, torch_FloatTensor_id);
  THFloatTensor *dispMapY_ptr = luaT_checkudata(L, 2, torch_FloatTensor_id);
  THFloatTensor *iL_ptr      = luaT_checkudata(L, 3, torch_FloatTensor_id);
  THFloatTensor *iR_ptr      = luaT_checkudata(L, 4, torch_FloatTensor_id);
  THFloatTensor *edges_ptr   = luaT_checkudata(L, 5, torch_FloatTensor_id);
  int corrWindowSize = lua_tonumber(L, 6);
  int dMin = lua_tonumber(L, 7);
  int dMax = lua_tonumber(L, 8);
  int UpDown = lua_tonumber(L,9);
  float th = lua_tonumber(L,10);

  //Debug - yes, it does work
  //printf("Test from C\n");

  // get raw pointers
  float *dispMapX = THFloatTensor_data(dispMapX_ptr);
  float *dispMapY = THFloatTensor_data(dispMapY_ptr);
  float *iL      = THFloatTensor_data(iL_ptr);
  float *iR      = THFloatTensor_data(iR_ptr);
  float *edges   = THFloatTensor_data(edges_ptr);

  // dims
  int nr = iL_ptr->size[0];
  int nc = iL_ptr->size[1];

  //strides
  long *is = iL_ptr->stride;
  long *os = dispMapX_ptr->stride;
  long *es = edges_ptr->stride;

  /*//Debug
    printf("is[0] = %li, is[1] = %li\n",is[0],is[1]);
    printf("os[0] = %li, os[1] = %li\n",os[0],os[1]);
    printf("es[0] = %li, es[1] = %li\n",es[0],es[1]);
    printf("th = %f\n", th);
    getchar();*/

  // stereo algoithm

  int i,j,ii,jj,d,ud;

  //    long k;
  //#pragma omp parallel for private(k)
  //    for (k = 0; k < 1000000000000; k++){
  //      long m = k*2+5;
  //      printf("%ld\n",m);
  //    }

int lowerLimit = 0;
if (dMin < 0) lowerLimit = -dMin;

#pragma omp parallel for private(i,j,ii,jj,d,ud)
  for (i = 0; i < nr-(corrWindowSize+2*UpDown-1); i++)
    for (j = lowerLimit; j < nc-(corrWindowSize+dMax-1); j++) {

      //Debug
      //printf("i = %i, j = %i\n",i,

      float prevCorrScore = 65532;
      int bestMatchSoFar = lowerLimit+dMin;
      int bestY = UpDown;


      if (*(edges+i*es[0]+j*es[1]) > th)
        for (ud = 0; ud <= 2*UpDown; ud++)
          for (d = dMin; d <= dMax; d++) {

            int pos0 = (i+UpDown)*is[0] + (j+dMin)*is[1];
            int pos1 = (i+ud)*is[0] + (j+dMin+d)*is[1];
            float corrScore = 0;

            for (ii = 0; ii < corrWindowSize; ii++) {
              float *i_ptr = iL+pos0+ii*is[0];
              float *j_ptr = iR+pos1+ii*is[0];
              for (jj = 0; jj < corrWindowSize; jj++) {
                //float pxDiff = abs( iL[pos0+ii*is[0]+jj*is[1] ] - iR[pos1+ii*is[0]+jj*is[1] ]);
                //float pxDiff = abs( *i_ptr  - iR[pos1+ii*is[0]+jj*is[1] ]);
                float pxDiff = abs( *i_ptr - *j_ptr );

                //Debug
                //printf("ii = %i, jj = %i, pxDiff = %f",ii,jj,pxDiff);

                corrScore += pxDiff;
                i_ptr++; j_ptr++;
              }
            }

            /*//Debug
              printf("corrScore = %f\n",corrScore);
              printf("prevCorrScore = %f\n",prevCorrScore);
              printf("bestMatchSoFar = %f, d = %i\n",d);
              getchar();*/
            /*printf("%d and ",UpDown);
            getchar();*/ //TODO check the 'memory'!!! it remembers!!

            if (corrScore < prevCorrScore) {
              prevCorrScore = corrScore;
              bestMatchSoFar = d;
              bestY = ud;
            }
          }

      dispMapY[ i*os[0]+(j-lowerLimit)*os[1] ] = bestY;
      dispMapX[ i*os[0]+(j-lowerLimit)*os[1] ] = bestMatchSoFar;
    }

  lua_newtable(L);           // result = {}
int result = lua_gettop(L);

return 0;
}

static const struct luaL_reg stereo_functions__  [] = {
  {"stereo",l_stereo},
  {NULL, NULL}
};

static void eex_StereoFunctions_init(lua_State *L)
{
  luaL_openlib(L,"eex",stereo_functions__,0);
  return 1;
}
