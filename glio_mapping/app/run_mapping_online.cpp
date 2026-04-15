#include "laser_mapping.h"
 
using namespace glio_mapping;
 
/**
 * @brief Main function - Entry point of the LaserMapping application
 * @param argc Number of command line arguments
 * @param argv Command line arguments
 * @return Exit code
 */
int main(int argc, char** argv)
{
    // Parse configuration path from command line arguments
    std::string config_path = "";
    if(argc >= 2){
        config_path = argv[1];
    }
    
    // Initialize ROS node
    ros::init(argc, argv, "laserMapping");
    
    // Create node handle
    ros::NodeHandle nh;
    
    // Create and initialize LaserMapping instance
    LaserMapping laser_mapping(nh, config_path);
    
    // Initialize the mapping system
    laser_mapping.init();
    
    // Run the main mapping loop
    laser_mapping.run();
    
    return 0;
}