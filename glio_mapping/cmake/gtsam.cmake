
find_package( GTSAMCMakeTools )
find_package(GTSAM REQUIRED QUIET)
list(APPEND ALL_TARGET_LIBRARIES gtsam)