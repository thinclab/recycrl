cmake_minimum_required(VERSION 3.8)
project(recycrl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Recyclables.srv"
  DEPENDENCIES geometry_msgs
)

# Need to use "chmod +x" before it is recognized as executable
install(
  PROGRAMS
    scripts/recycle.py
    scripts/imitate.py
    scripts/cam_align.py
    scripts/detect_service.py
    scripts/pnp.py
    scripts/image_capture.py
    scripts/yolo_service.py
    scripts/yolo_client.py
    scripts/train.py
    scripts/reset.py
    scripts/generate.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

ament_package()
