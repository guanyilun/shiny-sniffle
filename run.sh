##############
# lightcurve #
##############
# build lightcurve
# for tod in `cat mouse_f090.txt`;
# do
#     echo ${tod}
#     # python build_lightcurve.py --tod ${tod} -o out_mouse
#     python build_lightcurve.py --tod ${tod} -o out_saga \
#            --srcmask-tag s19_f090_gal_srcmask_saga
#     # python build_lightcurve.py --tod ${tod} -o out_mouse_w400 \
#     #        --window 400
# done
# visualize the readings
# python plot_lightcurve.py --idir out_mouse --oname lightcurve.png
# python plot_lightcurve.py --idir out_saga --oname lightcurve_saga.png
# python plot_lightcurve.py --idir out_mouse_p10 --oname lightcurve_p10.png

#################
# pulsar search #
#################
# idir=out_mouse_w400
# odir=out_w400
# python pulsar_search.py --infiles ${idir}/*ar5.npy -o $odir \
#        --freqs 5:20:500 --phis 0:360:50 --fwhm 0.01 --oname stats_ar5_f0d01.npy
# python pulsar_search.py --infiles ${idir}/*ar6.npy -o $odir \
    #        --freqs 5:20:500 --phis 0:360:50 --fwhm 0.01 --oname stats_ar6_f0d01.npy
# idir=out_saga
# odir=out
# python pulsar_search.py --infiles ${idir}/*ar6.npy -o $odir \
#        --freqs 1:20:51 --phis 0:360:100 --fwhm 0.01 --oname stats_ar6_f0d01.npy

##############
# simulation #
##############
# python sim_data.py -o out --data out_mouse/*ar5.npy \
#        --phi 40 --freq 8 --ampfrac 5 \
#        --oname sim_ar5_f12_p40_a1.npy --debug
# python pulsar_search.py --infiles out/sim_ar5_f12_p40_a1.npy \
#       --freqs 5:20:500 --phis 0:360:50 --fwhm 0.01 --oname stats_sim_f12_p40_a1.npy

#################################
# forced photometry from sigurd #
#################################
# OMP_NUM_THREADS=4 mpirun -n 10 python forced_photometry_subarray.py source_list.txt "s19,f090,night" out_photometry --dataset s19_galactic_new --sys gal

####################
# flux versus time #
####################
# python plot_flux_time.py -o plots --sid 0 --oname flux_saga.png --pwv-y 35 --dt-y 37 --xlim "[0,2000]"
# python plot_flux_time.py -o plots --sid 1 --oname flux_mouse.png --pwv-y 2.3 --dt-y 2.6
# python plot_flux_time.py -o plots --sid 2 --oname flux_s2.png --pwv-y 3 --dt-y 3.1
# python plot_flux_time.py -o plots --sid 3 --oname flux_s3.png --pwv-y 2.3 --dt-y 2.6


# 2021-11-23: start to seriously persue this project
python sim_data2.py 211123_source_list.txt "@211123_todlist.txt" out_test --dataset dr6v3

