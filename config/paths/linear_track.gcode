; Linear track replica of test_speed_spectral.py
; Velocity: 0.8 m/s = 800 mm/s = 48000 mm/min
; Start: X=0, Y=2.5 mm (0.0025m)
; Duration: 0.012 s -> Distance: 9.6 mm
; End: X=9.6, Y=2.5
; Power: 200 Watts

G21 ; Units in mm
G90 ; Absolute positioning

; Move to start position
G0 X0 Y2.5

; Turn laser on (M3 pseudo-code, S values map to Power)
M3 S200

; Linear move 
G1 X9.7 Y2.5 F48000

; Turn laser off
M5
