; Example 03 laser track (small box).
; Starts inside the domain (X = 0.2 mm) so the melt pool sits away from the walls,
; which makes the longitudinal (x-z) melt-pool section render cleanly.
; v_scan = 0.15 m/s = 9000 mm/min; y held at the domain centre (Ly/2 = 0.15 mm).
; Power S = 24 W; absorptivity is applied by the solver.

G21 ; Units in mm
G90 ; Absolute positioning

; Move to start position (laser off)
G0 X0.2 Y0.15

; Turn laser on, 24 W
M3 S24

; Linear move along +x at 9000 mm/min (0.15 m/s)
G1 X0.4 Y0.15 F9000

; Turn laser off
M5
