cmake_minimum_required(VERSION 3.22)
project(llamapp LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
find_package(CUDAToolkit REQUIRED)


# Ensure Clang is used as the CUDA host compiler
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang-14)

# Define source files
set(SOURCE_FILES
    src/models/llama.cu
    src/cu_utils/kernels.cu
    src/cu_utils/quantization.cu
    src/model_manager/gguf_loader.cpp
    src/model_manager/model_manager.cpp
    src/server/server.cpp
)

# Create executable
add_executable(llamapp ${SOURCE_FILES})
include_directories(include)

# Find and link required libraries
find_package(BLAS REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenSSL REQUIRED)
if(OPENSSL_FOUND)
    set(HTTPLIB_IS_USING_OPENSSL TRUE)
endif()

# Link libraries
set(CUDA_LIBS CUDA::cudart CUDA::cublas)

target_link_libraries(llamapp PRIVATE
    ${BLAS_LIBRARIES}
    ${CUDA_LIBS}
    nlohmann_json::nlohmann_json
    sentencepiece
    $<$<BOOL:${HTTPLIB_IS_USING_OPENSSL}>:OpenSSL::SSL>
    $<$<BOOL:${HTTPLIB_IS_USING_OPENSSL}>:OpenSSL::Crypto>
)

# Include directories
target_include_directories(llamapp PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Set compilation flags
target_compile_options(llamapp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --use_fast_math>)
