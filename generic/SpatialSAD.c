#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialSAD.c"
#else

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
  long weight_h = weight->size[2];
  long weight_w = weight->size[3];

  long nInputPlane = input->size[0];
  long input_h  = input->size[dimh];
  long input_w   = input->size[dimw];

  long output_h = (input_h - weight_h) / dH + 1;
  long output_w  = (input_w - weight_w) / dW + 1;

  THArgCheck(weight->size[1] == nInputPlane, 2, "invalid number of input planes");
  // TODO: check that # kernel channels is correct
  THArgCheck( (input_h >= weight_h && input_w >= weight_w) , 2, "SpatialSAD : Input image is smaller than kernel");

  if (input->nDimension == 3)
  {
    THTensor_(resize3d)(output, nOutputPlane, output_h, output_w);

    real *input_data = THTensor_(data)(input);
    real *weight_data = THTensor_(data)(weight);
    real *output_data = THTensor_(data)(output);

    // fill output with zero values
    long k;
    #pragma omp parallel for private(k)
    for (k = 0; k < nOutputPlane; k++)
    {
      real *ptr_output = output_data + k*output_w*output_h;
      long l;
      for (l = 0; l < output_h*output_w; l++)
        ptr_output[l] = 0.0;
    }

    long zz;
    #pragma omp parallel for private(k,zz)
    for(k = 0; k < nOutputPlane; k++)
    {
      for(zz = 0; zz < nInputPlane; zz++)
      {
        eex_(SAD)(output_data + k*output_w*output_h, output_h, output_w, 
                  input_data + zz*input_w*input_h, input_w, 
                  weight_data + k*nInputPlane*weight_w*weight_h + zz*weight_w*weight_h, weight_h, weight_w, 
                  dH, dW);
      }
    }
  }
  else
  {
    // TODO: rewrite batch mode
    //THTensor_(resize4d)(output, input->size[0], nOutputPlane, output_h, output_w);

  }

  THTensor_(free)(input);
  THTensor_(free)(weight);
  return 1;
}


static int nn_(SpatialSAD_updateGradInput)(lua_State *L)
{
  // TODO: Rewrite
  //THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
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
