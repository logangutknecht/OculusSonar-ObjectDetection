import struct
import numpy as np
import sys
from pathlib import Path

# Constants
MSG_HEADER_SIZE = 16
MSG_ID_SIMPLE_FIRE_V2 = 35
OCULUS_ID = 0x4F53  # 'OS'

def read_uint16(data, offset):
    return struct.unpack_from('<H', data, offset)[0]

def read_uint32(data, offset):
    return struct.unpack_from('<I', data, offset)[0]

def read_double(data, offset):
    return struct.unpack_from('<d', data, offset)[0]

def find_messages(raw_bytes):
    messages = []
    i = 0
    while i < len(raw_bytes) - MSG_HEADER_SIZE:
        oculus_id = read_uint16(raw_bytes, i)
        message_id = read_uint16(raw_bytes, i + 6)
        if oculus_id == OCULUS_ID and message_id == MSG_ID_SIMPLE_FIRE_V2:
            payload_size = read_uint32(raw_bytes, i + 10)
            full_size = MSG_HEADER_SIZE + payload_size
            if i + full_size > len(raw_bytes):
                break
            messages.append(raw_bytes[i:i+full_size])
            i += full_size
        else:
            i += 1
    return messages

def parse_fire_v2(msg_bytes):
    offset = 162    # Range resolution field (after fixed 161 bytes) - from docs
    range_res = struct.unpack_from('<d', msg_bytes, offset)[0]  # 8 bytes
    offset += 8                                                  # = 170
    range_count = struct.unpack_from('<H', msg_bytes, offset)[0] # 2 bytes
    offset += 2                                                  # = 172
    bearing_count = struct.unpack_from('<H', msg_bytes, offset)[0] # 2 bytes
    offset += 2                                                  # = 174
    offset += 4 * 4                                              # skip reserved[4] = 16 bytes → offset = 190
    image_offset = struct.unpack_from('<I', msg_bytes, offset)[0]
    offset += 4                                                  # 194
    image_size = struct.unpack_from('<I', msg_bytes, offset)[0]
    offset += 4                                                  # 198
    _ = struct.unpack_from('<I', msg_bytes, offset)[0]           # msgSize
    offset += 4                                                  # 202

    bearings = struct.unpack_from('<' + 'h' * bearing_count, msg_bytes, offset)
    offset += 2 * bearing_count

    # Payload
    image_data = msg_bytes[image_offset:image_offset + image_size]
    image = np.frombuffer(image_data, dtype=np.uint8).reshape((range_count, bearing_count))

    return {
        'bearing_count': bearing_count,
        'range_count': range_count,
        'range_resolution': range_res,
        'bearings': bearings,
        'image': image
    }

if __name__ == "__main__":
    # Use command-line argument if provided, else fallback to default
    file_path = sys.argv[1] if len(sys.argv) > 1 else "/home/devuser/ros2_ws/data/Oculus_20250705_150423.oculus"

    if not Path(file_path).is_file():
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    with open(file_path, "rb") as f:
        raw_data = f.read()

    packets = find_messages(raw_data)
    print(f"Found {len(packets)} SimpleFireV2 packets")

    if packets:
        parsed = parse_fire_v2(packets[0])
        print(f"Bearings: {parsed['bearing_count']}, Ranges: {parsed['range_count']}")
        print(f"Image shape: {parsed['image'].shape}")
