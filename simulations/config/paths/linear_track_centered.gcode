; Centered linear track for the temp_dep_316L domain (2 x 1 x 0.5 mm).
; y-centre = 0.5 mm (was 1.25 mm in linear_track.gcode -> outside this domain).
; Power 24 W (FE reference), scan 0.15 m/s = 150 mm/s = 9000 mm/min, start near x_min.
G21 ; Units in mm
G90 ; Absolute positioning
G0 X0.1 Y0.5        ; move to start (laser off)
M3 S24              ; laser on, 24 W
G1 X1.9 Y0.5 F9000  ; linear scan at 0.15 m/s
M5                  ; laser off
