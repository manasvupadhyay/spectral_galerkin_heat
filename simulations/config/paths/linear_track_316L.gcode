; FE-reference matching track (Theo_cuboid_FE_temp_dep.py)
; Start: X=0, Y=0.5 mm (domain centre of Ly=1.0 mm)
; v_scan = 0.15 m/s = 150 mm/s = 9000 mm/min
; Laser stays on for the whole run; at t=0.012 s the spot is at X=1.8 mm.
; Power S = 24 W (FE P_laser); absorptivity A=0.30 applied by the solver.

G21 ; Units in mm
G90 ; Absolute positioning

; Move to start position (laser off)
G0 X0.0 Y0.5

; Turn laser on, 24 W
M3 S24

; Linear move along +x at 9000 mm/min (0.15 m/s); run past 1.8 mm so the
; beam is on for the entire 0.012 s window.
G1 X2.0 Y0.5 F9000

; Turn laser off
M5
