import math
import argparse



def generate_ellipse_gcode(filename, cx, cy, rx, ry, num_points=300, feed_rate=48000, laser_power=200):
    # Calculate path distance and estimated time
    total_distance = 0.0
    points = []

    for i in range(num_points + 1):
        theta = 2 * math.pi * i / num_points * 4  # 4 loops around the ellipse
        x = cx + rx * (1 - i / num_points) * math.cos(theta)
        y = cy + ry * (1 - i / num_points) * math.sin(theta)
        points.append((x, y))

    for i in range(1, len(points)):
        distance = math.sqrt((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2)
        total_distance += distance

    # Time in seconds (feed_rate is in mm/min)
    time_seconds = (total_distance / feed_rate) * 60
    time_minutes = time_seconds / 60

    with open(filename, 'w') as f:
        f.write("; Ellipse Path\n")
        f.write(f"; Estimated total time: ({time_seconds:.5f} s)\n")
        f.write("; Turn laser on (M3 pseudo-code, S values map to Power)\n")
        f.write(f"; M3 S{laser_power} ; Power: {laser_power} Watts\n")
        f.write("G21 ; Set units to millimeters\n")

        for i, (x, y) in enumerate(points):
            if i == 0:
                f.write(f"G0 X{x:.3f} Y{y:.3f}\n")
                f.write(f"M3 S{laser_power}\n")
            else:
                f.write(f"G1 X{x:.3f} Y{y:.3f} F{feed_rate}\n")

        f.write("M5 ; Laser off\n")

if __name__ == '__main__':
    generate_ellipse_gcode('simulations/config/paths/ellipse.gcode', cx=1.25, cy=1.25, rx=0.9, ry=0.5, num_points=100)
    print("Generated ellipse.gcode")
