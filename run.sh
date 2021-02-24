##############
# lightcurve #
##############
# build lightcurve
for tod in `cat mouse_f090.txt`;
do
    echo ${tod}
    # python build_lightcurve.py --tod ${tod} -o out_mouse 
    python build_lightcurve.py --tod ${tod} -o out_mouse_w400 \
           --window 400
done
# visualize the readings
# python plot_lightcurve.py --idir out_mouse --oname lightcurve.png
# python plot_lightcurve.py --idir out_mouse_p10 --oname lightcurve_p10.png

#################
# pulsar search #
#################
# idir=out_mouse_p10
# odir=out_p10
# python pulsar_search.py --infiles ${idir}/*ar5.npy -o $odir \
#        --freqs 5:20:1000 --phis 0:360:50 --fwhm 0.01 --oname stats_ar5_f0d01.npy
# python pulsar_search.py --infiles ${idir}/*ar6.npy -o $odir \
#        --freqs 5:20:1000 --phis 0:360:50 --fwhm 0.01 --oname stats_ar6_f0d01.npy

##############
# simulation #
##############
# python sim_data.py -o out --data out_mouse/*ar5.npy \
#        --phi 40 --freq 12 --ampfrac 1 \
#        --oname sim_ar5_f12_p40_a1.npy --debug
# python pulsar_search.py --infiles out/sim_ar5_f12_p40_a1.npy \
#       --freqs 5:20:200 --phis 0:360:100 --fwhm 0.01 --oname stats_sim_f12_p40_a1.npy
