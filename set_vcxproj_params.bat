rem
rem This bat file sets the environment variables needed by
rem the vcxproj files used to build the CUDA examples
rem 
rem Please edit to correspond to your CUDA release version and GPU
rem CC level
rem
rem setx CUDA_SM   "compute_75,sm_75"
rem setx CUDA_SM   "compute_86,sm_86"
setx CUDA_SM   "compute_75,sm_75;compute_86,sm_86;"
setx CUDA_VER  "11.5"
setx CX_ROOT   "C:\new_code\inc"
rem
setx OpenCV_Root "D:\opencv454\build"
setx OpenCV_Lib  "opencv_world454.lib"
rem