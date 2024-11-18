# Target system details
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Set the size of void pointer (32-bit or 64-bit)
set(CMAKE_SIZEOF_VOID_P 4)  # 4 for 32-bit ARM, change to 8 for 64-bit

# Cross-compilers
set(CMAKE_C_COMPILER /../../../usr/bin/arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER /../../../usr/bin/arm-linux-gnueabihf-g++)

# Set sysroot to the Raspberry Pi root filesystem
set(CMAKE_SYSROOT /media/joshua/rootfs)

# Specify the target type to avoid running executables on host
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Enable cross-compiling mode
set(CMAKE_CROSSCOMPILING TRUE)

# Ensure the proper root path for libraries and includes
set(CMAKE_FIND_ROOT_PATH /media/joshua/rootfs)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Additional paths (if needed)
# set(OpenCV_INCLUDE_DIRS /media/joshua/rootfs/usr/local/include/opencv4)
# set(OpenCV_LIBRARIES /media/joshua/rootfs/usr/local/lib/libopencv_core.so
#                       /media/joshua/rootfs/usr/local/lib/libopencv_imgproc.so
#                       /media/joshua/rootfs/usr/local/lib/libopencv_highgui.so
#                       /media/joshua/rootfs/usr/local/lib/libopencv_videoio.so)
