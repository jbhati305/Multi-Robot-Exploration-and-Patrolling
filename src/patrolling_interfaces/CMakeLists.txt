cmake_minimum_required(VERSION 3.5)
project(patrolling_interfaces)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SendRobotToObject.srv"
)

# Export dependencies
ament_export_dependencies(rosidl_default_runtime)

# Package
ament_package()
