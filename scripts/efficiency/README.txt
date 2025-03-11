1. plot_efficiency_bz.py
   this read the data and plot the efficiency  for Big Z, Li, Be, Boron

2.plot_efficiency_tofq.py
   this read the histogram of data and mc, calculate data/mc efficiency corrections and plot the efficiency  for tofq, using Carbon

3.plot_efficiency_trigger.py
   this read the histogram of data and mc, calculate data/mc efficiency corrections and plot the efficiency  of trigger, using Carbon

4.plot_efficiency_inntrk.py
   this read the histogram of data and mc, calculate data/mc efficiency corrections and plot the efficiency  of inntrk, Li, Be, and Boron.
   The inntrk efficiency can only be estimated up to around 20 GV with ToF and Cutoff Rigidity, here one need to combine two histogram together(R_tof, R_cutoff)
   

4.plot_inntrkcharge_efficiency.py
   this read the histogram of data and mc, calculate data/mc efficiency corrections and plot the efficiency  of inntrk Z, Be

5. plot_efficiency_sedtrk_be_only.py
   this read the histogram of data and mc, calculate data/mc efficiency corrections and plot the efficiency  of second trk cut for Be

6. plot_efficiency_sedtrk.py
   this read the histogram of data and mc, calculate data/mc efficiency corrections and plot the efficiency  of second trk cut for He, Carbon, Be
   this cut efficiency correction is not able to estimate using Be, as data contains all the nuclei and MC is only Be, for He, the efficiency correction is close to 1,
   In Jiahui thesis: he use the He efficiency corrections as background is negiliable for He (probably?)
   
7.plot_efficiency_backgroundcut.py 
					
8. calculate_ql1charge_efficency_parametrization.py
    parameterize the charge template and then fit to data of L1 charge

9. compute_rich_efficiency.py
   read data and MC root file, fill the events before and after pass the RICH cuts
   calculate the efficiency and store as np

10. plot_rich_efficiency.py
   read the npz results from compute_rich_efficiency.py, plot the rich efficiency and comparisons with Jiahui 

10. compute_tof_efficiency.py
   read data and MC root file, fill the events before and after pass the RICH cuts
   calculate the efficiency and store as np

11. plot_template_charge.py
    plot the template of charge
    

  plot_efficiency_total.py


