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
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${DBoW2_INCLUDE_DIRS}
)

# add_executable(${PROJECT_NAME}_node src/main.cpp)
add_executable(${PROJECT_NAME}_dbow2 src/main_dbow2.cpp
                                     src/DBoW2/BowVector.cpp
                                     src/DBoW2/FBrief.cpp
                                     src/DBoW2/FeatureVector.cpp
                                     src/DBoW2/FORB.cpp
                                     src/DBoW2/FSurf64.cpp
                                     src/DBoW2/QueryResults.cpp
                                     src/DBoW2/ScoringObject.cpp)

add_executable(${PROJECT_NAME}_dbow3 src/main_dbow3.cpp
                                     src/DBoW3/BowVector.cpp
                                     src/DBoW3/Database.cpp
                                     src/DBoW3/DescManip.cpp
                                     src/DBoW3/FeatureVector.cpp
                                     src/DBoW3/QueryResults.cpp
                                     src/DBoW3/quicklz.c
                                     src/DBoW3/ScoringObject.cpp
                                     src/DBoW3/Vocabulary.cpp)

target_link_libraries(${PROJECT_NAME}_dbow2
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${DBoW2_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME}_dbow3
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${DBoW2_LIBRARIES}
)