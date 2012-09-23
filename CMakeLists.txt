
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(OpenMP)

IF (OPENMP_FOUND)
  MESSAGE (STATUS "OpenMP Found with compiler flag : ${OpenMP_C_FLAGS}")
  SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
ENDIF (OPENMP_FOUND)

SET(src init.c)
FILE(GLOB luasrc *.lua)
FILE(GLOB testsrc test/*.lua)
SET(luasrc ${luasrc} ${testsrc})
#SET(luasrc ${luasrc} test/test-modules.lua)
ADD_TORCH_PACKAGE(eex "${src}" "${luasrc}" "e-Lab extras")
TARGET_LINK_LIBRARIES(eex luaT TH)
