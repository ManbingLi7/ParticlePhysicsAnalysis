1. generate_masshist_counts_mc.py
   generate 2d histograms for MC, including mass vs Ekn, Rigidity and beta resdiual/resolution
   choose conditions (Pass8 GBL, use tuned beta and fine/Rebin binning)

2. calculate_rich_acceptance.py (get the counts of MC -> calculate and plot the acceptance)

2. generate_mass_hist_iss.py
   generate 2d histogram for ISS Data

# make_jiahui_mc2dhistogram.py: read the root files from jiahui and make a 2D histogram for the mass_function_fit_jiahuimc.py

3. 

How to get the fluxes ingredients:
1). ISS 2D mass hist
2). MC 2D mass hist -> MC template fits -> get PValues
3). MC study energy loss with Truth Info ( 2D hist) -> mean value of delta (1/m) -> k_mu
4). MC Mass resolution -> k_\sigma
5) Template fit data -> counts
6) Acceptances: run the script in lxplus: calculate_totnum.py to get the total number of generated events, run script calculate_acceptance.py



4. make_beta_eventsatHighR.py: for MC tuning study, run root to generate 1d hist of events with High R



3. 
   5.1plot_template_mass.py
   function fit Aerogel MC: Be7 Be9 Be10 mass distribution bin by bin indepedently and compare the tem_shape_pars to validate the scaling factor for different mass templates
   ->scaling factor for width(sigma) is not exactly m7/m9 for example, need to be studied
   5.2 plot_template_mass_be7.py
   function fit MC Be7 plot the fit parameters, fit the tem_shape_pars with poly function
   plot_template_mass_be7_Agl.py, plot_template_mass_be7_NaF.py, plot_template_mass_be7_Tof.py
   Seperate the three detectors as they need different way of constraint for the fit parameters
   
   

9. study_rigidity_resolution.py (this can be removed)
   step1: input 2d histogram of isotopes, (eg. Be7, Be9, Be10)
   step2: fit resolution bin by bin for each isotopes
   step3: plot the width(sigma) of the fit of Be7/9/10 in the sample figure.
   step4: plot the width ratio of 7/9, 7/10 (factor to be added in the scaling factor in mass template)
   
10. make_rigidity_resolution_histogram.py
    produce the 2d histogram of rigidity resolution as function of true Ekin/n

10. make_resolution_histogram.py
    python3 make_resolution_histogram.py --nuclei Be
    produce the 2d histogram of rigidity resolution  and beta resolution as function of true Ekin/n


14. be_massfit_mcvalidation.py
    use the tuned MC from Jiahui
    1) fit the Be7 MC, all parameters free
    2) fit the Be7 MC, set limit to parameters accroding to the first fit
    3) fit the Be7 MC, simultaneously.
    4) fit the Be9, Be10MC bin by bin, get the scaling factor in MC(need some constraint in the parameters
    5) fit the Be9, Be10MC simultaneously, get the scaling factor in MC.
    6) mixture the Be7, Be9, Be10MC with constant mixture value(0.6, 0.3, 0.1), using the test dataset
    7) use the model to fit the mixed MC, to get the output event counts.
    8) mixture the Be7, Be9, Be10MC with some energy dependence, or for example, use the value I get from data.
    9) use the model to fit the mixed MC(energy dependence mixture).

15. Study the rigidity resolution.
    be_rigfit_allsteps.py "17" means using the true rigidity at LowerToF/RICH (0:L1, 7:L2)
    python3 be_rigfit_allsteps.py --filename_mc /home/manbing/Documents/Data/data_rig/rigidity_resolution_17.npz --plotdir plots/rigfit_17 


16. Study the resolution :
    beta_residual_fit.py
    


17. plot_efficiency_total.py
    this script reads all the efficiency from /home/manbing/Documents/Data/data_effacc/eff_corrections which was stored as spline
    and plot all efficiency together and calculate the total efficiency
    

18 generate_masshist_cutoff.py
   
