cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

set(CMAKE_C_FLAGS_RELEASE "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O4 -DNDEBUG")

include_directories("../")

add_library(vednn_c_code OBJECT
	vednnConvolutionForward.cpp
#	vednnConvolutionForwardAddBias.c
	vednnConvolutionBackwardData.cpp
	vednnConvolutionBackwardFilter.cpp
	vednnLinearForward.cpp
	vednnLinearBackwardData.cpp
	vednnLinearBackwardWeight.cpp
#	vednnActivationForward.c
#	vednnActivationBackward.c
#	vednnMaxPoolingForward.c
#	vednnMaxPoolingForward_default.c
#	vednnMaxPoolingBackward.c
#	vednnMaxPoolingBackward_default.c
#	vednnSoftmaxForward.c
	vednnInit.c
)

if(USE_OPENMP)
	target_compile_options(vednn_c_code PUBLIC -fopenmp)
	target_compile_definitions(vednn_c_code PUBLIC -DVEDNN_USE_OPENMP)
endif()

if(BUILD_SHARED_LIB)
	target_compile_options(vednn_c_code PUBLIC -fpic)
endif()