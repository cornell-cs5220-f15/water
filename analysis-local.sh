for seq_num in {0..8}
do
        amplxe-cl -report hotspots -r r00${seq_num}ah > analysis_0$(($seq_num+1))
done

amplxe-cl -report hotspots -r r009ah > analysis_10

for seq_num in {10..14}
do
        amplxe-cl -report hotspots -r r0${seq_num}ah > analysis_$(($seq_num+1))
done
