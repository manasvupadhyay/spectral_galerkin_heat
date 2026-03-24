; Start: X=0, Y=2.5 mm (0.0025m)
; Duration: 0.012 s -> Distance: 9.6 mm
; End: X=9.6, Y=2.5


G21 ; Units in mm
G90 ; Absolute positioning

; Move to start position
G0 X0.0 Y1.25

; Turn laser on (M3 pseudo-code, S values map to Power)
M3 S200 ; Power: 200 Watts

; Linear move 
G1 X9.7 Y1.25 F48000 ; Velocity: 0.8 m/s = 800 mm/s = 48000 mm/min

; Turn laser off
M5
