add_library(${PROJECT_NAME}
        imu_processing.cpp
        laser_mapping.cpp
        preprocess.cpp
        ../include/ikd-Tree/ikd_Tree.cpp 
        )

target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${OpenMP_CXX_FLAGS}
        )

target_include_directories(${PROJECT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})

