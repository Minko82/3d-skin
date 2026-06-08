import serial
import struct
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation



SERIAL_PORT = '/dev/cu.usbmodem1101'
BAUD_RATE = 115200
NUM_SENSORS = 3

STRUCT_FORMAT = '<' + ('ffff' * NUM_SENSORS) 
PACKET_SIZE = struct.calcsize(STRUCT_FORMAT)

MAX_HISTORY = 100 


history_x = [collections.deque([0]*MAX_HISTORY, maxlen=MAX_HISTORY) for _ in range(NUM_SENSORS)]
history_y = [collections.deque([0]*MAX_HISTORY, maxlen=MAX_HISTORY) for _ in range(NUM_SENSORS)]
history_z = [collections.deque([0]*MAX_HISTORY, maxlen=MAX_HISTORY) for _ in range(NUM_SENSORS)]

baselines_x = [None] * NUM_SENSORS
baselines_y = [None] * NUM_SENSORS
baselines_z = [None] * NUM_SENSORS

fig, axes = plt.subplots(NUM_SENSORS, 1, figsize=(10, 3 * NUM_SENSORS), sharex=True)
fig.canvas.manager.set_window_title('Magnetic Skin Data')
fig.tight_layout(pad=3.0)

if NUM_SENSORS == 1:
    axes = [axes]

lines_x, lines_y, lines_z = [], [], []

for i in range(NUM_SENSORS):
    ax = axes[i]
    ax.set_title(f"Sensor {i+1}")
    ax.set_ylabel("Delta")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    lx, = ax.plot([], [], label='Δ X', color='#ff4c4c', lw=2)
    ly, = ax.plot([], [], label='Δ Y', color='#4cff4c', lw=2)
    lz, = ax.plot([], [], label='Δ Z', color='#4c4cff', lw=2)
    ax.legend(loc='upper left')
    
    lines_x.append(lx)
    lines_y.append(ly)
    lines_z.append(lz)
axes[-1].set_xlabel("Samples")


try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    ser.reset_input_buffer()
    print(f"Connected to {SERIAL_PORT}. Expecting {NUM_SENSORS} sensors.")
    print("Waiting for valid data to capture baselines...")
except Exception as e:
    print(f"Failed to connect: {e}")
    exit()

def update_plot(frame):
    if ser.in_waiting > PACKET_SIZE * 5:
        ser.reset_input_buffer()

    while ser.in_waiting >= PACKET_SIZE + 2:
        if ser.read(1) == b'\xAA':
            if ser.read(1) == b'\xBB':
                raw_data = ser.read(PACKET_SIZE)
                
                if len(raw_data) == PACKET_SIZE:
                    unpacked = struct.unpack(STRUCT_FORMAT, raw_data)
                    
                    for i in range(NUM_SENSORS):
                        offset = i * 4 
                        x, y, z = unpacked[offset + 1], unpacked[offset + 2], unpacked[offset + 3]
                        
                        if x < 78000.0 and y < 78000.0 and z < 126000.0:
                            
                            if baselines_x[i] is None:
                                baselines_x[i] = x
                                baselines_y[i] = y
                                baselines_z[i] = z
                                print(f"Sensor {i} Baseline Locked -> X:{x:.1f}  Y:{y:.1f}  Z:{z:.1f}")

                            dx = x - baselines_x[i]
                            dy = y - baselines_y[i]
                            dz = z - baselines_z[i]

                            history_x[i].append(dx)
                            history_y[i].append(dy)
                            history_z[i].append(dz)


    x_axis = range(MAX_HISTORY)
    for i in range(NUM_SENSORS):
        lines_x[i].set_data(x_axis, history_x[i])
        lines_y[i].set_data(x_axis, history_y[i])
        lines_z[i].set_data(x_axis, history_z[i])
        
        c_min = min(min(history_x[i]), min(history_y[i]), min(history_z[i]))
        c_max = max(max(history_x[i]), max(history_y[i]), max(history_z[i]))
        
        max_swing = max(abs(c_min), abs(c_max))
        pad = max_swing * 0.1
        if max_swing == 0: pad = 10
        
        axes[i].set_ylim(-max_swing - pad, max_swing + pad)
        axes[i].set_xlim(0, MAX_HISTORY)

    return lines_x + lines_y + lines_z

ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    ser.close()
    print("Closed connection.")
