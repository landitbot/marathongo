add_executable(${PROJECT_NAME}_node run_mapping_online.cpp)
target_compile_options(${PROJECT_NAME}_node PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME}_node
        ${PROJECT_NAME} gflags thirdparty
        ${OpenMP_CXX_FLAGS}
        )