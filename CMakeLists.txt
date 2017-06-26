cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)
project( Vanish )
find_package( OpenCV REQUIRED)
find_package( Armadillo REQUIRED)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${ARMADILLO_INCLUDE_DIRS}
  )
add_executable( vanish main.cpp )
target_link_libraries(vanish
  ${OpenCV_LIBS}
  ${ARMADILLO_LIBRARIES}
)
