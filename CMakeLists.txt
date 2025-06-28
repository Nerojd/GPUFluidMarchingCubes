cmake_minimum_required(VERSION 3.20)

project(GPUFluidMarchingCubes LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)

# Add the linker option only for MSVC 
if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()

# === GLAD source ===
set(GLAD_SRC ${CMAKE_SOURCE_DIR}/dependencies/glad/src/glad.c)

# === GLFW source ===
add_subdirectory(dependencies/glfw)

# === GLUT source ===
# add_subdirectory(dependencies/freeglut)

# === Main sources ===
file(GLOB SOURCES
    ${CMAKE_SOURCE_DIR}/main.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cu
    ${CMAKE_SOURCE_DIR}/shaders/*.cpp
)
# set(SOURCES
#     ${CMAKE_SOURCE_DIR}/main.cpp
#     ${CMAKE_SOURCE_DIR}/src/Grid.cu
#     ${CMAKE_SOURCE_DIR}/src/MarchingCubes.cu
#     ${CMAKE_SOURCE_DIR}/src/ParticleSystem.cu
#     ${CMAKE_SOURCE_DIR}/src/Renderer.cpp
#     ${CMAKE_SOURCE_DIR}/shaders/shaders.cpp
# )

# === OpenGL ===
find_package(OpenGL REQUIRED)

# === Executable ===
add_executable(${CMAKE_PROJECT_NAME}
    ${SOURCES}
    ${GLAD_SRC}
)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: > #-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored > #--expt-relaxed_constexpr> #-g -G
)

# === Includes ===
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/dependencies/CUDA
    ${CMAKE_SOURCE_DIR}/dependencies/glad/include
    ${CMAKE_SOURCE_DIR}/dependencies/glfw/include
    ${CMAKE_SOURCE_DIR}/dependencies/freeglut/include
    ${CMAKE_SOURCE_DIR}/dependencies/glew/include
    ${CMAKE_SOURCE_DIR}/dependencies/glm
    ${CMAKE_SOURCE_DIR}/include
)

# === Linking ===
target_link_libraries(${CMAKE_PROJECT_NAME}
    OpenGL::GL
    glfw
    freeglut
    ${CMAKE_SOURCE_DIR}/dependencies/glew/lib/Release/Win32/glew32.lib
    ${CMAKE_SOURCE_DIR}/dependencies/glew/lib/Release/x64/glew32.lib
    # -lGL -lGLU -lglut -lGLEW -lcuda -lcudart -lcudaGL -lfreeglut -lopengl32 -lglu32
)

if(MSVC)
    target_link_options(${CMAKE_PROJECT_NAME} PRIVATE /NODEFAULTLIB:LIBCMT)
endif()