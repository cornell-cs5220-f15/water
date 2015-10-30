set terminal x11
set key at 3.25,29
#set key width 1
set key horizontal
set title "Bubble Case - Rough Time Comparison"
set xtic auto
set ytic auto
set xlabel "Simulation Time (s)"
#set xr [0.0:30.0]
set ylabel "Wall Time (s)"
#set yr [-12.0:12.0]

set style line 1 lt 1 lw 2 pt 8 ps 0.75 linecolor rgb "blue"
set style line 2 lt 1 lw 2 pt 4 ps 0.75 linecolor rgb "red"
set style line 3 lt 1 lw 2 pt 6 ps 0.75 linecolor rgb "green"
set style line 4 lt 1 lw 2 pt 6 ps 0.75 linecolor rgb "cyan"
plot "ACLS/monitor/timing" every ::1 u 2:9 t 'ACLS: MP Step' w l ls 3, \
"VOF/monitor/timing" every :: 1 u 2:9 t  'VOF: MP Step' w l ls 1, \
"ACLS/monitor/timing" every ::1 u 2:3 t 'ACLS: Total Time' w l ls 2, \
"VOF/monitor/timing" every ::1 u 2:3 t 'VOF: Total Time' w l ls 4


################################################################################

set term png
set output "ACLS_VOF_time.png"
replot
set term x11
