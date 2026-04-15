import struct
from collections import deque


class SBUSParser:
    def __init__(self):
        # 25字节的数据帧
        # [0]: 帧头 (0x0F)
        # [1-22]: 16个通道数据 (每个通道11bit)
        # [23]: 标志位 (包含 Ch17, Ch18, 信号丢失, 故障保护)
        # [24]: 帧尾 (0x00)
        self.channels = [0] * 18
        self.failsafe = False
        self.frame_lost = False

    def decode(self, frame):
        """
        解析25字节的SBUS原始数据帧
        :param frame: bytes 类型，长度为 25
        :return: bool 是否解析成功
        """
        if len(frame) != 25:
            return False

        if frame[0] != 0x0F:
            return False

        # 提取通道数据 (每11 bits一个通道)
        # 这部分通过位移操作将22字节拼接成16个11bit通道
        self.channels[0] = ((frame[1] | frame[2] << 8) & 0x07FF)
        self.channels[1] = ((frame[2] >> 3 | frame[3] << 5) & 0x07FF)
        self.channels[2] = ((frame[3] >> 6 | frame[4] <<
                            2 | frame[5] << 10) & 0x07FF)
        self.channels[3] = ((frame[5] >> 1 | frame[6] << 7) & 0x07FF)
        self.channels[4] = ((frame[6] >> 4 | frame[7] << 4) & 0x07FF)
        self.channels[5] = ((frame[7] >> 7 | frame[8] <<
                            1 | frame[9] << 9) & 0x07FF)
        self.channels[6] = ((frame[9] >> 2 | frame[10] << 6) & 0x07FF)
        self.channels[7] = ((frame[10] >> 5 | frame[11] << 3) & 0x07FF)
        self.channels[8] = ((frame[12] | frame[13] << 8) & 0x07FF)
        self.channels[9] = ((frame[13] >> 3 | frame[14] << 5) & 0x07FF)
        self.channels[10] = (
            (frame[14] >> 6 | frame[15] << 2 | frame[16] << 10) & 0x07FF)
        self.channels[11] = ((frame[16] >> 1 | frame[17] << 7) & 0x07FF)
        self.channels[12] = ((frame[17] >> 4 | frame[18] << 4) & 0x07FF)
        self.channels[13] = (
            (frame[18] >> 7 | frame[19] << 1 | frame[20] << 9) & 0x07FF)
        self.channels[14] = ((frame[20] >> 2 | frame[21] << 6) & 0x07FF)
        self.channels[15] = ((frame[21] >> 5 | frame[22] << 3) & 0x07FF)

        # 标志位解析 [23]
        flags = frame[23]
        self.channels[16] = 1 if flags & 0x01 else 0  # Ch17 (Digital)
        self.channels[17] = 1 if flags & 0x02 else 0  # Ch18 (Digital)
        self.frame_lost = bool(flags & 0x04)        # 信号丢失
        self.failsafe = bool(flags & 0x08)        # 故障保护激活

        return True

    def get_channels(self):
        return self.channels

    def __str__(self):
        return f"Channels: {self.channels[:8]}... FS: {self.failsafe}"


class SBUSReceiver:
    def __init__(self):
        self.buffer = deque(maxlen=25)
        self.parser = SBUSParser()

    def process_byte(self, new_byte):
        # 如果传入的是 int，直接添加；如果是 bytes，取第一个
        val = new_byte[0] if isinstance(
            new_byte, (bytes, bytearray)) else new_byte
        self.buffer.append(val)

        # 只有满25字节才尝试解析
        if len(self.buffer) == 25:
            # 校验帧头
            if self.buffer[0] == 0x0F:
                # 某些设备帧尾不一定是0x00，可以根据实际情况放宽限制
                # if self.buffer[24] == 0x00:
                success = self.parser.decode(bytes(self.buffer))
                if success:
                    # 解析成功，清空缓冲区等待下一帧，防止滑动窗口误触发
                    self.buffer.clear()
                    return True
            # 如果帧头不对或者解析失败，deque会自动在下次append时弹出最旧的字节
        return False

"""
import serial
ser = serial.Serial('/dev/ttyAMA0', 100000, parity='N', stopbits=1)
receiver = SBUSReceiver()
while True:
    byte = ser.read(1)
    if receiver.process_byte(byte):
        print(receiver.parser.channels)
"""
