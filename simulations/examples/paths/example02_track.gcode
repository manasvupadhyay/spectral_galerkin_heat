; Example 02 track (small box, constant 316L + latent heat).
; Small domain so a fine z-grid resolves the ~few-micron mushy zone cheaply.
; v_scan = 0.4 m/s (F24000 mm/min); y at the domain centre (Ly/2 = 0.2 mm);
; P = 120 W (S120), absorptivity applied by the solver.

G21 ; Units in mm
G90 ; Absolute positioning

; Move to start position (laser off)
G0 X0.15 Y0.2

; Turn laser on, 120 W
M3 S120

; Linear move along +x at 24000 mm/min (0.4 m/s)
G1 X0.5 Y0.2 F24000

; Turn laser off
M5
