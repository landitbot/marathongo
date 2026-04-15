SET(LIB_FILE
    loop/euclidean_loop.cpp
    back/pose_graph_manager.cc
    gtsam_solver/gtsam_solver.cpp
    estimator/glio_estimator.cpp
)

add_library(thirdparty
        ${LIB_FILE}
        )

target_compile_options(thirdparty PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(thirdparty
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${PYTHON_LIBRARIES}
        tbb
        glog
        ${ALL_TARGET_LIBRARIES}
        yaml-cpp
        ${OpenMP_CXX_FLAGS}
        )

target_include_directories(thirdparty PRIVATE ${PYTHON_INCLUDE_DIRS})


