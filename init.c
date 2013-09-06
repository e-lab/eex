#include "TH.h"
#include "luaT.h"
//#include "omp.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)
#define eex_(NAME) TH_CONCAT_3(eex_, Real, NAME)

#include "generic/sad.c"
#include "THGenerateFloatTypes.h"
#include "generic/SpatialSAD.c"
#include "THGenerateFloatTypes.h"
#include "generic/SpatialSADMap.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialMaxMap.c"
#include "THGenerateFloatTypes.h"
#include "generic/SpatialSADMap.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialMaxMap.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_libeex(lua_State *L)
{
  nn_FloatSpatialSAD_init(L);
  nn_FloatSpatialSADMap_init(L);
  nn_FloatSpatialMaxMap_init(L);

  nn_DoubleSpatialSAD_init(L);
  nn_DoubleSpatialSADMap_init(L);
  nn_DoubleSpatialMaxMap_init(L);

  return 1;
}
