"""ROS2 message definition strings for MCAP schema registration."""

UINT16_MULTIARRAY_SCHEMA = b"""\
# std_msgs/msg/UInt16MultiArray
MultiArrayLayout  layout
uint16[]          data

================================================================================
MSG: std_msgs/MultiArrayLayout
MultiArrayDimension[] dim
uint32 data_offset

================================================================================
MSG: std_msgs/MultiArrayDimension
string label
uint32 size
uint32 stride
"""

FLOAT32_MULTIARRAY_SCHEMA = b"""\
# std_msgs/msg/Float32MultiArray
MultiArrayLayout  layout
float32[]         data

================================================================================
MSG: std_msgs/MultiArrayLayout
MultiArrayDimension[] dim
uint32 data_offset

================================================================================
MSG: std_msgs/MultiArrayDimension
string label
uint32 size
uint32 stride
"""

POINTCLOUD2_SCHEMA = b"""\
# sensor_msgs/msg/PointCloud2
std_msgs/Header header
uint32 height
uint32 width
PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: sensor_msgs/PointField
string name
uint32 offset
uint8 datatype
uint32 count
"""

TFMESSAGE_SCHEMA = b"""\
# tf2_msgs/msg/TFMessage
geometry_msgs/TransformStamped[] transforms

================================================================================
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec

================================================================================
MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation

================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
"""
