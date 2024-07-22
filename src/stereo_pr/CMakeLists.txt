cmake_minimum_required(VERSION 3.0.2)
project(stereo_pr)

set(OpenCV_DIR /home/cona/open_library/opencv-3.4.16/install/share/OpenCV)
# set(DBoW2_DIR /home/cona/open_library/DBoW2/install/lib/cmake/DBoW2)

# list(APPEND CMAKE_PREFIX_PATH ${DBoW2_DIR})

find_package(catkin REQUIRED COMPONENTS
  roscpp
)
find_package(OpenCV REQUIRED)
# find_package(DBoW2 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES stereo_pr
  CATKIN_DEPENDS roscpp
  DEPENDS system_lib
)

include_directories(
  include
  include/DBoW2
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${DBoW2_INCLUDE_DIRS}
)

# add_executable(${PROJECT_NAME}_node src/main.cpp)
add_executable(${PROJECT_NAME}_node src/main.cpp
                                    src/DBoW2/BowVector.cpp
                                    src/DBoW2/FBrief.cpp
                                    src/DBoW2/FeatureVector.cpp
                                    src/DBoW2/FORB.cpp
                                    src/DBoW2/FSurf64.cpp
                                    src/DBoW2/QueryResults.cpp
                                    src/DBoW2/ScoringObject.cpp)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${DBoW2_LIBRARIES}
)