from fast_heat_solv.core.laser import LaserPath, LaserState
import numpy as np

class GCodeLaserPath(LaserPath):
    def __init__(self, gcode_file: str, initial_position=(0.0, 0.0)):
        self.segments, self.unit_scale = self._parse_gcode(gcode_file)
        # Store initial position as float32 (scaled to meters)
        self.initial_position = (
            np.float32(initial_position[0]) * np.float32(self.unit_scale),
            np.float32(initial_position[1]) * np.float32(self.unit_scale)
        )
        self.current_segment = 0

    def _parse_gcode(self, filepath):
        # Parses G0 (move), G1 (linear cut), M3/M5 for power, and G21 for units
        segments = []
        current_pos = [0.0, 0.0]
        current_power = 0.0
        is_on = False
        feedrate = 0.0  # mm/min
        t = 0.0
        last_pos = None
        last_t = 0.0
        unit_scale = 1.0  # Default: mm (will convert to meters)
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                if line.startswith('G21'):
                    # Set units to mm, but solver expects meters, so scale = 1/1000
                    unit_scale = 1.0 / 1000.0
                    continue
                if line.startswith('G20'):
                    # Set units to inches, but solver expects meters, so scale = 25.4/1000
                    unit_scale = 25.4 / 1000.0
                    continue
                if line.startswith('G0') or line.startswith('G1'):
                    # Extract X, Y, F
                    tokens = line.split()
                    x = y = None
                    for token in tokens:
                        if token.startswith('X'):
                                    x = np.float32(float(token[1:]) * unit_scale)
                        elif token.startswith('Y'):
                                    y = np.float32(float(token[1:]) * unit_scale)
                        elif token.startswith('F'):
                                    feedrate = np.float32(float(token[1:]) * unit_scale)  # mm/min
                    if x is not None:
                        current_pos[0] = float(x)
                    if y is not None:
                        current_pos[1] = float(y)
                    if last_pos is not None and is_on:
                        # Calculate distance and time
                        dx = current_pos[0] - last_pos[0]
                        dy = current_pos[1] - last_pos[1]
                        dist = float(np.sqrt(dx*dx + dy*dy))
                        # feedrate is mm/min, convert to mm/s
                        speed = float(feedrate) / 60.0 if feedrate > 0 else 1.0
                        dt = dist / speed if speed > 0 else 0.0
                        t1 = float(last_t + dt)
                        # Ensure numeric values inside segments are float32 where appropriate
                        seg_start = (np.float32(last_pos[0]), np.float32(last_pos[1]))
                        seg_end = (np.float32(current_pos[0]), np.float32(current_pos[1]))
                        segments.append((seg_start, seg_end, np.float32(last_t), np.float32(t1), np.float32(current_power), bool(is_on)))
                        last_t = t1
                    last_pos = list(current_pos)
                elif line.startswith('M3'):
                    # Laser on, extract S (power)
                    tokens = line.split()
                    for token in tokens:
                        if token.startswith('S'):
                            current_power = float(token[1:])
                    is_on = True
                elif line.startswith('M5'):
                    # Laser off
                    is_on = False
                    current_power = 0.0
        return segments, unit_scale

    def get_state(self, time: float, dt: float) -> LaserState:
        # Find the segment for the given time
        for seg in self.segments:
            start_pos, end_pos, t0, t1, power, is_on = seg
            if t0 <= time <= t1:
                # Linear interpolation
                frac = (time - t0) / (t1 - t0) if t1 > t0 else 0.0
                x = np.float32(start_pos[0] + frac * (end_pos[0] - start_pos[0]))
                y = np.float32(start_pos[1] + frac * (end_pos[1] - start_pos[1]))
                # Compute velocity using previous position and time
                if time > t0 and dt > 0.0:
                    prev_frac = ((time - dt) - t0) / (t1 - t0) if (time - dt) >= t0 and (t1 - t0) > 0 else 0.0
                    prev_x = start_pos[0] + prev_frac * (end_pos[0] - start_pos[0])
                    prev_y = start_pos[1] + prev_frac * (end_pos[1] - start_pos[1])
                    vx = np.float32((float(x) - float(prev_x)) / float(dt))
                    vy = np.float32((float(y) - float(prev_y)) / float(dt))
                else:
                    vx = np.float32(0.0)
                    vy = np.float32(0.0)
                return LaserState(x=np.float32(x), y=np.float32(y), power=np.float32(power), is_on=is_on, v=(vx, vy))
        
        # If time is past the last segment, remain at the final position
        if self.segments and time > self.segments[-1][3]:
             end_pos = self.segments[-1][1]
             # Laser off after path ends (ensure float32)
             return LaserState(x=np.float32(end_pos[0]), y=np.float32(end_pos[1]), power=np.float32(0.0), is_on=False, v=(np.float32(0.0), np.float32(0.0)))

        # If time is before first segment or otherwise unmatched, use initial position
        return LaserState(x=np.float32(self.initial_position[0]), y=np.float32(self.initial_position[1]), power=np.float32(0.0), is_on=False, v=(np.float32(0.0), np.float32(0.0)))
