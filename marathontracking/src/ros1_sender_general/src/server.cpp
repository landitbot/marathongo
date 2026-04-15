#include <arpa/inet.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <netinet/in.h>
#include <ros/ros.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "std_srvs/Trigger.h"

// 帧头和帧尾定义（4字节）
constexpr uint32_t FRAME_HEADER = 0xA5A5A5A5;
constexpr uint32_t FRAME_TAIL = 0x5A5A5A5A;

constexpr uint32_t CMD_CAT1_MODE = 0x11;
constexpr uint32_t CMD_CAT1_ACTION = 0xAA;

constexpr uint32_t CMD_MODE_INIT_RECEIVER = 0b10101010;
constexpr uint32_t CMD_MODE_ZERO_TORQUE = 0b10101010 << 1;
constexpr uint32_t CMD_MODE_DAMPING = 0b10101010 << 2;
constexpr uint32_t CMD_MODE_WALKING = 0b10101010 << 3;
constexpr uint32_t CMD_MODE_RUNNING = 0b10101010 << 4;
constexpr uint32_t CMD_MODE_RC0_MODE = 0b10101010 << 5;
constexpr uint32_t CMD_MODE_RC1_MODE = 0b10101010 << 6;
constexpr uint32_t CMD_MODE_POWEROFF = 0xA1B2C3D4;

// 消息类型
enum MsgType : uint8_t {
  MSGTYPE_CMD_VEL = 0x01,
  MSGTYPE_HEARTBEAT = 0x02,
  MSGTYPE_CMD = 0x03,
};

enum JOYMODE { MODE_MOTION_READY = 1, MODE_DUMPING = 2, MODE_START = 3 };

// 客户端地址信息
struct ClientInfo {
  sockaddr_in addr;
  time_t last_heartbeat;
};

std::vector<ClientInfo> clients;
std::mutex clients_mutex;
int sockfd;

// 计算CRC32校验值
uint32_t calculate_crc32(const uint8_t* data, size_t length) {
  static const uint32_t crc_table[256] = {
      0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F,
      0xE963A535, 0x9E6495A3, 0x0EDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988,
      0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91, 0x1DB71064, 0x6AB020F2,
      0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
      0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9,
      0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
      0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C,
      0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
      0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423,
      0xCFBA9599, 0xB8BDA50F, 0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
      0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D, 0x76DC4190, 0x01DB7106,
      0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
      0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D,
      0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
      0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950,
      0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
      0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7,
      0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
      0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9, 0x5005713C, 0x270241AA,
      0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
      0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81,
      0xB7BD5C3B, 0xC0BA6CAD, 0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A,
      0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683, 0xE3630B12, 0x94643B84,
      0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
      0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB,
      0x196C3671, 0x6E6B06E7, 0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC,
      0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5, 0xD6D6A3E8, 0xA1D1937E,
      0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
      0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55,
      0x316E8EEF, 0x4669BE79, 0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236,
      0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F, 0xC5BA3BBE, 0xB2BD0B28,
      0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
      0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F,
      0x72076785, 0x05005713, 0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38,
      0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21, 0x86D3D2D4, 0xF1D4E242,
      0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
      0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69,
      0x616BFFD3, 0x166CCF45, 0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2,
      0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB, 0xAED16A4A, 0xD9D65ADC,
      0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
      0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD70693,
      0x54DE5729, 0x23D967BF, 0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
      0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D};
  uint32_t crc = 0xFFFFFFFF;

  for (size_t i = 0; i < length; ++i) {
    crc = (crc >> 8) ^ crc_table[(crc ^ data[i]) & 0xFF];
  }

  return crc ^ 0xFFFFFFFF;
}

// 处理客户端心跳
void handle_heartbeat() {
  uint8_t buffer[1024];
  sockaddr_in client_addr;
  socklen_t client_addr_len = sizeof(client_addr);

  while (ros::ok()) {
    // 非阻塞接收心跳包
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(sockfd, &readfds);

    timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000;  // 100ms

    int activity = select(sockfd + 1, &readfds, nullptr, nullptr, &timeout);
    if (activity <= 0) continue;

    memset(buffer, 0, sizeof(buffer));
    int recv_len = recvfrom(sockfd, buffer, sizeof(buffer), 0,
                            (struct sockaddr*)&client_addr, &client_addr_len);
    if (recv_len >= 16) {  // 最小心跳包长度
      // 解析心跳包
      uint32_t header = ntohl(*((uint32_t*)buffer));
      uint32_t tail = ntohl(*((uint32_t*)(buffer + recv_len - 4)));
      uint8_t msg_type = buffer[8];
      uint32_t data_length = ntohl(*((uint32_t*)(buffer + 4)));

      if (header == FRAME_HEADER && tail == FRAME_TAIL &&
          msg_type == MSGTYPE_HEARTBEAT && data_length == 1) {
        // 计算CRC校验
        uint32_t received_crc = ntohl(*((uint32_t*)(buffer + 9)));
        uint32_t calculated_crc = calculate_crc32(buffer + 8, 1);

        if (received_crc == calculated_crc) {
          std::lock_guard<std::mutex> lock(clients_mutex);

          // 检查客户端是否已存在
          bool found = false;
          for (auto& client : clients) {
            if (client.addr.sin_addr.s_addr == client_addr.sin_addr.s_addr &&
                client.addr.sin_port == client_addr.sin_port) {
              client.last_heartbeat = time(nullptr);
              found = true;
              break;
            }
          }

          // 新客户端加入
          if (!found) {
            ClientInfo new_client;
            new_client.addr = client_addr;
            new_client.last_heartbeat = time(nullptr);
            clients.push_back(new_client);
            ROS_INFO("New Client Connected: %s:%d, Current Client Num: %zu",
                     inet_ntoa(client_addr.sin_addr),
                     ntohs(client_addr.sin_port), clients.size());
          }
        }
      }
    }
  }
}

// 发送速度指令（带帧头帧尾和校验）
void send_cmd_vel_stamped(const geometry_msgs::TwistStamped::ConstPtr& msg) {
  // 协议格式: [4B帧头] [4B数据长度] [1B消息类型] [速度数据] [4B CRC32校验]
  // [4B帧尾]
  const size_t data_size = 1 + sizeof(float) * 6;  // 1B消息类型 + 6个float值
  const size_t packet_size =
      4 + 4 + data_size + 4 + 4;  // 帧头+长度+数据+校验+帧尾
  uint8_t buffer[packet_size];

  // 1. 设置帧头
  *((uint32_t*)buffer) = htonl(FRAME_HEADER);

  // 2. 设置数据长度
  *((uint32_t*)(buffer + 4)) = htonl(data_size);

  // 3. 设置消息类型
  buffer[8] = MSGTYPE_CMD_VEL;
  float vel_cmd[6];
  vel_cmd[0] = msg->twist.linear.x;
  vel_cmd[1] = msg->twist.linear.y;
  vel_cmd[2] = msg->twist.linear.z;
  vel_cmd[3] = msg->twist.angular.x;
  vel_cmd[4] = msg->twist.angular.y;
  vel_cmd[5] = msg->twist.angular.z;
  memcpy(buffer + 9, vel_cmd, sizeof(vel_cmd));

  // 5. 计算并设置CRC32校验（针对消息类型+数据部分）
  uint32_t crc = calculate_crc32(buffer + 8, data_size);
  *((uint32_t*)(buffer + 8 + data_size)) = htonl(crc);

  // 6. 设置帧尾
  *((uint32_t*)(buffer + 8 + data_size + 4)) = htonl(FRAME_TAIL);

  // 发送数据给所有客户端
  std::lock_guard<std::mutex> lock(clients_mutex);

  // 移除超时的客户端
  time_t current_time = time(nullptr);
  clients.erase(std::remove_if(clients.begin(), clients.end(),
                               [current_time](const ClientInfo& client) {
                                 return (current_time - client.last_heartbeat) >
                                        3;
                               }),
                clients.end());

  // 发送数据
  for (const auto& client : clients) {
    sendto(sockfd, buffer, packet_size, 0, (struct sockaddr*)&client.addr,
           sizeof(client.addr));
  }
}

// 发送速度指令（带帧头帧尾和校验）
void send_cmd_vel(const geometry_msgs::Twist::ConstPtr& msg) {
  // 协议格式: [4B帧头] [4B数据长度] [1B消息类型] [速度数据] [4B CRC32校验]
  // [4B帧尾]
  const size_t data_size = 1 + sizeof(float) * 6;  // 1B消息类型 + 6个float值
  const size_t packet_size =
      4 + 4 + data_size + 4 + 4;  // 帧头+长度+数据+校验+帧尾
  uint8_t buffer[packet_size];

  // 1. 设置帧头
  *((uint32_t*)buffer) = htonl(FRAME_HEADER);

  // 2. 设置数据长度
  *((uint32_t*)(buffer + 4)) = htonl(data_size);

  // 3. 设置消息类型
  buffer[8] = MSGTYPE_CMD_VEL;
  float vel_cmd[6];
  vel_cmd[0] = msg->linear.x;
  vel_cmd[1] = msg->linear.y;
  vel_cmd[2] = msg->linear.z;
  vel_cmd[3] = msg->angular.x;
  vel_cmd[4] = msg->angular.y;
  vel_cmd[5] = msg->angular.z;
  memcpy(buffer + 9, vel_cmd, sizeof(vel_cmd));

  // 5. 计算并设置CRC32校验（针对消息类型+数据部分）
  uint32_t crc = calculate_crc32(buffer + 8, data_size);
  *((uint32_t*)(buffer + 8 + data_size)) = htonl(crc);

  // 6. 设置帧尾
  *((uint32_t*)(buffer + 8 + data_size + 4)) = htonl(FRAME_TAIL);

  // 发送数据给所有客户端
  std::lock_guard<std::mutex> lock(clients_mutex);

  // 移除超时的客户端
  time_t current_time = time(nullptr);
  clients.erase(std::remove_if(clients.begin(), clients.end(),
                               [current_time](const ClientInfo& client) {
                                 return (current_time - client.last_heartbeat) >
                                        3;
                               }),
                clients.end());

  // 发送数据
  for (const auto& client : clients) {
    sendto(sockfd, buffer, packet_size, 0, (struct sockaddr*)&client.addr,
           sizeof(client.addr));
  }
}

void send_event(uint8_t msg_type, uint32_t params[6]) {
  // 这里虽然发送了6个float，但是在接收端没有用到
  // 仅作为保留使用，看以后是否会有使用到初始化参数

  // 协议格式: [4B帧头] [4B数据长度] [1B消息类型] [速度数据] [4B CRC32校验]
  // [4B帧尾]
  const size_t data_size = 1 + sizeof(float) * 6;  // 1B消息类型 + 6个float值
  const size_t packet_size =
      4 + 4 + data_size + 4 + 4;  // 帧头+长度+数据+校验+帧尾
  uint8_t buffer[packet_size];

  // 1. 设置帧头
  *((uint32_t*)buffer) = htonl(FRAME_HEADER);

  // 2. 设置数据长度
  *((uint32_t*)(buffer + 4)) = htonl(data_size);

  // 3. 设置消息类型
  buffer[8] = msg_type;
  uint32_t vel_cmd[6];
  vel_cmd[0] = params[0];  // reserved
  vel_cmd[1] = params[1];  // reserved
  vel_cmd[2] = params[2];  // reserved
  vel_cmd[3] = params[3];  // reserved
  vel_cmd[4] = params[4];  // reserved
  vel_cmd[5] = params[5];  // reserved
  memcpy(buffer + 9, vel_cmd, sizeof(vel_cmd));

  // 5. 计算并设置CRC32校验（针对消息类型+数据部分）
  uint32_t crc = calculate_crc32(buffer + 8, data_size);
  *((uint32_t*)(buffer + 8 + data_size)) = htonl(crc);

  // 6. 设置帧尾
  *((uint32_t*)(buffer + 8 + data_size + 4)) = htonl(FRAME_TAIL);

  // 发送数据给所有客户端
  std::lock_guard<std::mutex> glock(clients_mutex);

  // 移除超时的客户端
  time_t current_time = time(nullptr);
  clients.erase(std::remove_if(clients.begin(), clients.end(),
                               [current_time](const ClientInfo& client) {
                                 return (current_time - client.last_heartbeat) >
                                        3;
                               }),
                clients.end());

  // 发送数据
  // 支持服务端多次发送，客户端会有加限制，只会初始化一次
  for (const auto& client : clients) {
    sendto(sockfd, buffer, packet_size, 0, (struct sockaddr*)&client.addr,
           sizeof(client.addr));
  }
}

bool srv_handler_init_udp_receiver(std_srvs::Trigger::Request& req,
                                   std_srvs::Trigger::Response& res) {
  uint32_t params[6] = {CMD_CAT1_MODE, CMD_MODE_INIT_RECEIVER, 0, 0, 0, 0};
  send_event(MSGTYPE_CMD, params);
  res.success = 1;
  res.message = "success";
  return true;
}

bool srv_handler_zero_torque(std_srvs::Trigger::Request& req,
                             std_srvs::Trigger::Response& res) {
  uint32_t params[6] = {CMD_CAT1_MODE, CMD_MODE_ZERO_TORQUE, 0, 0, 0, 0};
  send_event(MSGTYPE_CMD, params);
  res.success = 1;
  res.message = "success";
  return true;
}

bool srv_handler_damping(std_srvs::Trigger::Request& req,
                         std_srvs::Trigger::Response& res) {
  uint32_t params[6] = {CMD_CAT1_MODE, CMD_MODE_DAMPING, 0, 0, 0, 0};
  send_event(MSGTYPE_CMD, params);
  res.success = 1;
  res.message = "success";
  return true;
}

bool srv_handler_walking(std_srvs::Trigger::Request& req,
                         std_srvs::Trigger::Response& res) {
  uint32_t params[6] = {CMD_CAT1_MODE, CMD_MODE_WALKING, 0, 0, 0, 0};
  send_event(MSGTYPE_CMD, params);
  res.success = 1;
  res.message = "success";
  return true;
}

bool srv_handler_running(std_srvs::Trigger::Request& req,
                         std_srvs::Trigger::Response& res) {
  uint32_t params[6] = {CMD_CAT1_MODE, CMD_MODE_RUNNING, 0, 0, 0, 0};
  send_event(MSGTYPE_CMD, params);
  res.success = 1;
  res.message = "success";
  return true;
}

bool srv_handler_poweroff(std_srvs::Trigger::Request& req,
                          std_srvs::Trigger::Response& res) {
  uint32_t params[6] = {CMD_CAT1_MODE, CMD_MODE_POWEROFF, 0, 0, 0, 0};
  send_event(MSGTYPE_CMD, params);
  res.success = 1;
  res.message = "success";
  return true;
}

bool srv_handler_rc0_mode_receiver(std_srvs::Trigger::Request& req,
                                   std_srvs::Trigger::Response& res) {
  uint32_t params[6] = {CMD_CAT1_MODE, CMD_MODE_RC0_MODE, 0, 0, 0, 0};
  send_event(MSGTYPE_CMD, params);
  res.success = 1;
  res.message = "success";
  return true;
}

bool srv_handler_rc1_mode_receiver(std_srvs::Trigger::Request& req,
                                   std_srvs::Trigger::Response& res) {
  uint32_t params[6] = {CMD_CAT1_MODE, CMD_MODE_RC1_MODE, 0, 0, 0, 0};
  send_event(MSGTYPE_CMD, params);
  res.success = 1;
  res.message = "success";
  return true;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "udp_cmd_vel_server");
  ros::NodeHandle nh;

  // 创建UDP套接字
  sockfd = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd < 0) {
    ROS_FATAL("Socket Create Failed");
    return 1;
  }

  // 设置服务器地址
  sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  std::string local_ip = "0.0.0.0";
  int local_port = 8888;
  server_addr.sin_addr.s_addr = inet_addr(local_ip.c_str());
  server_addr.sin_port = htons(static_cast<uint16_t>(local_port));
  if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    ROS_FATAL("Bind Failed");
    close(sockfd);
    return 1;
  }
  ROS_INFO("UDP SERVER Started, listening: %s:%d", local_ip.c_str(),
           local_port);

  std::string topic_name_stamped = "/final_stampd_cmd_vel";
  ros::Subscriber sub_cmd_vel_stamp =
      nh.subscribe(topic_name_stamped, 1, send_cmd_vel_stamped);

  // std::string topic_name = "/final_cmd_vel";
  // ros::Subscriber sub_cmd_vel = nh.subscribe(topic_name, 1, send_cmd_vel);

  // std::string topic_name3 = "/cmd_vel_to_send";
  // ros::Subscriber sub_cmd_vel3 = nh.subscribe(topic_name3, 1, send_cmd_vel);

  // 用于发送INIT类型数据，用于初始化机器人端
  auto srv_init_robot_recevier_2 = nh.advertiseService(
      "/cmd_server/init_udp_receiver", srv_handler_init_udp_receiver);

  auto srv_init_robot_recevier =
      nh.advertiseService("/init_udp_receiver", srv_handler_init_udp_receiver);

  auto srv_zero_torque =
      nh.advertiseService("/cmd_server/zero_torque", srv_handler_zero_torque);

  auto srv_damping =
      nh.advertiseService("/cmd_server/damping", srv_handler_damping);

  auto srv_walking =
      nh.advertiseService("/cmd_server/walking", srv_handler_walking);

  auto srv_running =
      nh.advertiseService("/cmd_server/running", srv_handler_running);

  auto srv_poweroff =
      nh.advertiseService("/cmd_server/poweroff", srv_handler_poweroff);

  auto srv_third_rc0_mode = nh.advertiseService(
      "/cmd_server/rc0_mode", srv_handler_rc0_mode_receiver);

  auto srv_third_rc1_mode = nh.advertiseService(
      "/cmd_server/rc1_mode", srv_handler_rc1_mode_receiver);

  // 创建线程处理心跳
  std::thread heartbeat_thread(handle_heartbeat);

  // 进入ROS循环
  ros::spin();

  // 关闭资源
  heartbeat_thread.detach();
  close(sockfd);

  return 0;
}