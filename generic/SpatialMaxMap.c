#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMaxMap.c"
#else

// TODO: implement backprop, save indices of max value
static int nn_(SpatialMaxMap_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  //THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_Tensor);

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");
  luaL_argcheck(L, input->size[0] == nInputPlane, 2, "invalid number of input planes");

  THTensor_(resize3d)(output, nOutputPlane,
                      input->size[1],
                      input->size[2]);

  // contiguous
  input = THTensor_(newContiguous)(input);
  output = THTensor_(newContiguous)(output);

  // get raw pointers
  real *input_data = THTensor_(data)(input);
  real *output_data = THTensor_(data)(output);
  real *connTable_data = THTensor_(data)(connTable);

  // and dims
  long input_h = input->size[1];
  long input_w = input->size[2];
  long output_h = output->size[1];
  long output_w = output->size[2];

  long p;
#pragma omp parallel for private(p)
  for (p = 0; p < nOutputPlane; p++) {
    // fill output with -inf
    real *ptr_output = output_data + p*output_w*output_h;
    long i,j,k;
    for(j = 0; j < output_h*output_w; j++)
      ptr_output[j] = -THInf;

    int nweight = connTable->size[0];
    for (k = 0; k < nweight; k++) {
      // get offsets for input/output
      int o_idx = (int)connTable_data[k*2+1]-1;
      int i_idx = (int)connTable_data[k*2+0]-1;

      if (o_idx == p) {
        real *input_p = input_data + k*input_w*input_h;
        real *output_p = ptr_output;

        // loop over input plane
        for(i = 0; i < input_h; i++)
        {
          for(j = 0; j < input_w; j++)
          {
            // local pointers
            //real *indyp = indy_p + k*owidth*oheight + i*owidth + j;
            //real *indxp = indx_p + k*owidth*oheight + i*owidth + j;

            // compute local max:
            //long maxindex = i_idx;
            real maxval = *output_p;
            real val = *input_p;
            if (val > maxval)
            {
              maxval = val;
            }

            // set output to local max
            *output_p = maxval;

            // store location of max (x,y)
            //*indyp = (int)(maxindex / kW)+1;
            //*indxp = (maxindex % kW) +1;

            input_p++;
            output_p++;
          }
        }
      }
    }
  }

  // clean up
  THTensor_(free)(input);
  THTensor_(free)(output);

  return 1;
}

// TODO: implement backprop, updateGradInput
static int nn_(SpatialMaxMap_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
//  int dW = luaT_getfieldcheckint(L, 1, "dW");
//  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
//
//  // contiguous
//  gradInput = THTensor_(newContiguous)(gradInput);
//  gradOutput = THTensor_(newContiguous)(gradOutput);
//
//  // Resize/Zero
//  THTensor_(resizeAs)(gradInput, input);
//  THTensor_(zero)(gradInput);
//
//  // get raw pointers
//  real *gradInput_data = THTensor_(data)(gradInput);
//  real *gradOutput_data = THTensor_(data)(gradOutput);
//  real *weight_data = THTensor_(data)(weight);
//  real *connTable_data = THTensor_(data)(connTable);
//
//  // and dims
//  long input_h = input->size[1];
//  long input_w = input->size[2];
//  long output_h = gradOutput->size[1];
//  long output_w = gradOutput->size[2];
//  long weight_h = weight->size[1];
//  long weight_w = weight->size[2];
//
//  long p;
//#pragma omp parallel for private(p)
//  for(p = 0; p < nInputPlane; p++)
//    {
//      long k;
//      // backward all
//      int nkernel = connTable->size[0];
//      for(k = 0; k < nkernel; k++)
//        {
//          int o = (int)connTable_data[k*2+1]-1;
//          int i = (int)connTable_data[k*2+0]-1;
//          if (i == p)
//            {
//              // gradient to input
//              THTensor_(fullConv2Dptr)(gradInput_data + i*input_w*input_h,
//                                    1.0,
//                                    gradOutput_data + o*output_w*output_h,  output_h,  output_w,
//                                    weight_data + k*weight_w*weight_h, weight_h, weight_w,
//                                    dH, dW);
//            }
//        }
//    }
//
//  // clean up
//  THTensor_(free)(gradInput);
//  THTensor_(free)(gradOutput);
//
  return 1;
}


static const struct luaL_Reg nn_(SpatialMaxMap__) [] = {
  {"SpatialMaxMap_updateOutput", nn_(SpatialMaxMap_updateOutput)},
  {"SpatialMaxMap_updateGradInput", nn_(SpatialMaxMap_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SpatialMaxMap_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialMaxMap__), "nn");
  lua_pop(L,1);
}

#endif
