#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialSADMap.c"
#else

static int nn_(SpatialSADMap_updateOutput)(lua_State *L)
{
 THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_Tensor);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  //THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");
  luaL_argcheck(L, input->size[0] == nInputPlane, 2, "invalid number of input planes");
  luaL_argcheck(L, input->size[2] >= kW && input->size[1] >= kH, 2, "input image smaller than kernel size");

  THTensor_(resize3d)(output, nOutputPlane,
                      (input->size[1] - kH) / dH + 1,
                      (input->size[2] - kW) / dW + 1);

  // contiguous
  input = THTensor_(newContiguous)(input);
  output = THTensor_(newContiguous)(output);

  // get raw pointers
  real *input_data = THTensor_(data)(input);
  real *output_data = THTensor_(data)(output);
  real *weight_data = THTensor_(data)(weight);
  //real *bias_data = THTensor_(data)(bias);
  real *connTable_data = THTensor_(data)(connTable);

  // and dims
  const long input_h = input->size[1];
  const long input_w = input->size[2];
  const long output_h = output->size[1];
  const long output_w = output->size[2];
  const long weight_h = weight->size[1];
  const long weight_w = weight->size[2];

  // fill output with zero values
  long p;
//#pragma omp parallel for private(k)
  for (p = 0; p < nOutputPlane; p++)
  {
    real *ptr_output = output_data + p*output_w*output_h;
    long l;
    for (l = 0; l < output_h*output_w; l++)
      ptr_output[l] = 0.0;
  }
  
  int k;
  int nweight = connTable->size[0];
//#pragma omp parallel for private(p)
  for (p = 0; p < nOutputPlane; p++) {
    // convolve all maps
    for (k = 0; k < nweight; k++) {
      // get offsets for input/output
      int o = (int)connTable_data[k*2+1]-1;
      int i = (int)connTable_data[k*2+0]-1;

      if (o == p)
        {
          eex_(SAD)(output_data + o*output_w*output_h, output_h, output_w, 
                    input_data + i*input_w*input_h, input_w, 
                    weight_data + k*weight_w*weight_h, weight_h, weight_w, 
                    dH, dW);
        }
    }
  }

  // clean up
  THTensor_(free)(input);
  THTensor_(free)(output);

  return 1;
}

static int nn_(SpatialSADMap_updateGradInput)(lua_State *L)
{
//  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
//  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
//  int dW = luaT_getfieldcheckint(L, 1, "dW");
//  int dH = luaT_getfieldcheckint(L, 1, "dH");
//  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
//
//  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_Tensor);
//  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
//  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
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
  luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  return 1;
}

static int nn_(SpatialSADMap_accGradParameters)(lua_State *L)
{
//  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
//  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
//  int dW = luaT_getfieldcheckint(L, 1, "dW");
//  int dH = luaT_getfieldcheckint(L, 1, "dH");
//  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
//  real scale = luaL_optnumber(L, 4, 1);
//
//  THTensor *connTable = luaT_getfieldcheckudata(L, 1, "connTable", torch_Tensor);
//  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
//  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
//  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
//
//  // contiguous
//  input = THTensor_(newContiguous)(input);
//  gradOutput = THTensor_(newContiguous)(gradOutput);
//
//  // get raw pointers
//  real *input_data = THTensor_(data)(input);
//  real *gradOutput_data = THTensor_(data)(gradOutput);
//  real *gradWeight_data = THTensor_(data)(gradWeight);
//  real *gradBias_data = THTensor_(data)(gradBias);
//
//  // and dims
//  long input_h = input->size[1];
//  long input_w = input->size[2];
//  long output_h = gradOutput->size[1];
//  long output_w = gradOutput->size[2];
//  long weight_h = weight->size[1];
//  long weight_w = weight->size[2];
//
//  // gradients wrt bias
//  long k;
//#pragma omp parallel for private(k)
//  for(k = 0; k < nOutputPlane; k++) {
//    real *ptr_gradOutput = gradOutput_data + k*output_w*output_h;
//    long l;
//    for(l = 0; l < output_h*output_w; l++)
//      gradBias_data[k] += scale*ptr_gradOutput[l];
//  }
//
//  // gradients wrt weight
//  int nkernel = connTable->size[0];
//#pragma omp parallel for private(k)
//  for(k = 0; k < nkernel; k++)
//    {
//      int o = (int)THTensor_(get2d)(connTable,k,1)-1;
//      int i = (int)THTensor_(get2d)(connTable,k,0)-1;
//
//      // gradient to kernel
//      THTensor_(validXCorr2DRevptr)(gradWeight_data + k*weight_w*weight_h,
//                                 scale,
//                                 input_data + i*input_w*input_h, input_h, input_w,
//                                 gradOutput_data + o*output_w*output_h, output_h, output_w,
//                                 dH, dW);
//    }
//
//  // clean up
//  THTensor_(free)(input);
//  THTensor_(free)(gradOutput);
  return 0;
}

static const struct luaL_Reg nn_(SpatialSADMap__) [] = {
  {"SpatialSADMap_updateOutput", nn_(SpatialSADMap_updateOutput)},
  {"SpatialSADMap_updateGradInput", nn_(SpatialSADMap_updateGradInput)},
  {"SpatialSADMap_accGradParameters", nn_(SpatialSADMap_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SpatialSADMap_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialSADMap__), "nn");
  lua_pop(L,1);
}

#endif
