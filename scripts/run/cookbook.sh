# add corrected LIP beta 
#python3 add_correctedlipbeta.py --filename /home/manbing/Documents/Data/ISS/BeISS_B1130P7_CMATCor.root --nuclei Be

#python3 add_treebranch_mc.py --filename /home/manbing/Documents/Data/MC/Be7MC_B1220l1_NucleiSelection.root --nuclei Be --dataname MC &
#python3 add_treebranch_mc.py --filename /home/manbing/Documents/Data/MC/Be9MC_B1220l1_NucleiSelection.root --nuclei Be --isotope Be9 --dataname MC &
#python3 add_treebranch_mc.py --filename /home/manbing/Documents/Data/MC/Be10MC_B1220l1_NucleiSelection.root --nuclei Be --isotope Be10 --dataname MC &
#wait

#select TOF tree of MC
python3 ../write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be7MC_BetaCor.root --betatype CIEMAT --detector Tof --nuclei Be --isotope Be7 --qsel 4.0  --outputprefix Be7MC_Tof_CMAT_V1 --resultdir ../trees/MC --treename amstreea --datatype MC

wait
python3 ../write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be9MC_BetaCor.root --betatype CIEMAT --detector Tof --nuclei Be --isotope Be9 --qsel 4.0  --outputprefix Be9MC_Tof_CMAT_V1 --resultdir ../trees/MC --treename amstreea --datatype MC

wait
python3 ../write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be10MC_BetaCor.root --betatype CIEMAT --detector Tof --nuclei Be --isotope Be10 --qsel 4.0  --outputprefix Be10MC_Tof_CMAT_V1 --resultdir ../trees/MC --treename amstreea --datatype MC

#select RICH-Agl tree of MC
python3 ../write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be7MC_BetaCor.root --betatype CIEMAT --detector Agl --nuclei Be --isotope Be7 --qsel 4.0 --isIss False --outputprefix Be7MC_RICHAgl_CMAT_V2 --resultdir ../trees/MC2 --treename amstreea --datatype MC 
wait
python3 ../write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be9MC_BetaCor.root --betatype CIEMAT --detector Agl --nuclei Be --isotope Be9 --qsel 4.0 --isIss False --outputprefix Be9MC_RICHAgl_CMAT_V2 --resultdir ../trees/MC2 --treename amstreea --datatype MC   
wait
python3 ../write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be10MC_BetaCor.root --betatype CIEMAT --detector Agl --nuclei Be --isotope Be10 --qsel 4.0 --isIss False --outputprefix Be10MC_RICHAgl_CMAT_V2 --resultdir ../trees/MC2 --treename amstreea --datatype MC   
wait

#select RICH-NaF tree of MC
python3 ../write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be7MC_BetaCor.root --betatype CIEMAT --detector NaF --nuclei Be --isotope Be7 --qsel 4.0 --isIss False --outputprefix Be7MC_RICHNaF_CMAT_V2 --resultdir ../trees/MC2 --treename amstreea --datatype MC   
wait
python3 ../write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be9MC_BetaCor.root  --betatype CIEMAT --detector NaF --nuclei Be --isotope Be9 --qsel 4.0 --isIss False --outputprefix Be9MC_RICHNaF_CMAT_V2 --resultdir ../trees/MC2 --treename amstreea --datatype MC   
wait
python3 ../write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be10MC_BetaCor.root  --betatype CIEMAT --detector NaF --nuclei Be --isotope Be10 --qsel 4.0 --isIss False --outputprefix Be10MC_RICHNaF_CMAT_V2 --resultdir ../trees/MC2 --treename amstreea --datatype MC   
#wait

#counts be7Iss tof
#python3 write_tree_q.py --filename trees/ISS/BeISS_BetaCor.root --treename amstreea_cor  --nuclei Be --isotope Be7 --qsel 4.0 --isIss 1 --outputprefix Be7ISS_Tof_V1 --resultdir trees/ISS --detector Tof --betatype CIEMAT --datatype ISS
#wait
#python3 create_mass2dhist_V2.py --filename trees/ISS/Be7ISS_Tof_V1_0.root --treename amstreea_cor --isotope Be7 --detector Tof
#wait
#root /home/manbing/Documents/LithiumAnalysis_RWTH/Scripts/Be_Analysis_V2/combinedFit_Sys_main_Be7.C

#counts be9Iss tof
#python3 write_tree_q.py --filename trees/ISS/BeISS_BetaCor.root --treename amstreea_cor  --nuclei Be --isotope Be9 --qsel 4.0 --isIss 1 --outputprefix Be9ISS_Tof_V1 --resultdir trees/ISS --detector Tof --betatype CIEMAT --datatype ISS  
#wait
#python3 create_mass2dhist_V2.py --filename trees/ISS/Be9ISS_Tof_V1_0.root --treename amstreea_cor --isotope Be9 --detector Tof
#wait 
#python3 write_tree_q.py --filename trees/ISS/BeISS_BetaCor.root --treename amstreea_cor  --nuclei Be --isotope Be10 --qsel 4.0 --isIss 1 --outputprefix Be10ISS_Tof_V1 --resultdir trees/ISS --detector Tof --betatype CIEMAT
#wait
#python3 create_mass2dhist_V2.py --filename trees/ISS/Be10ISS_Tof_V1_0.root --treename amstreea_cor --isotope Be10 --detector Tof

#counts be7Iss agl
#python3 write_tree_q.py --filename trees/ISS/BeISS_BetaCor.root --treename amstreea_cor  --nuclei Be --isotope Be7 --qsel 4.0 --isIss 1 --outputprefix Be7ISS_RICHAgl_V1 --resultdir trees/ISS --detector Agl --betatype CIEMAT --datatype ISS  
#wait
#python3 create_mass2dhist_V2.py --filename trees/ISS/Be7ISS_RICHAgl_V1_0.root --treename amstreea_cor --isotope Be7 --detector Agl
#wait

#counts be9Iss agl
#python3 write_tree_q.py --filename trees/ISS/BeISS_BetaCor.root --treename amstreea_cor  --nuclei Be --isotope Be9 --qsel 4.0 --isIss 1  --outputprefix Be9ISS_RICHAgl_V1 --resultdir trees/ISS --detector Agl --datatype ISS --betatype CIEMAT
#wait
#python3 create_mass2dhist_V2.py --filename trees/ISS/Be9ISS_RICHAgl_V1_0.root --treename amstreea_cor --isotope Be9
#wait

#counts be10
#python3 write_tree_q.py --filename trees/ISS/BeISS_BetaCor.root --treename amstreea_cor  --nuclei Be --isotope Be10 --qsel 4.0  --outputprefix Be10ISS_RICHAgl_V1 --resultdir trees/ISS --datatype ISS --betatype CIEMAT --datatype ISS   
#wait
#python3 create_mass2dhist_V2.py --filename trees/ISS/Be10ISS_RICHAgl_V1_0.root --treename amstreea_cor --isotope Be10
#wait

#counts be7 NaF
#python3 write_tree_q.py --filename trees/ISS/BeISS_BetaCor.root --treename amstreea_cor  --nuclei Be --isotope Be7 --qsel 4.0  --outputprefix Be7ISS_RICHNaF_V1 --resultdir trees/ISS --detector NaF --betatype CIEMAT --datatype ISS
#wait
#python3 create_mass2dhist_V2.py --filename trees/ISS/Be7ISS_RICHNaF_V1_0.root --treename amstreea_cor --isotope Be7 --detector NaF
#wait

#counts be9 NaF
#python3 write_tree_q.py --filename trees/ISS/BeISS_BetaCor.root --treename amstreea_cor  --nuclei Be --isotope Be9 --qsel 4.0 --isIss 1  --outputprefix Be9ISS_RICHNaF_V1 --resultdir trees/ISS --detector NaF --betatype CIEMAT --datatype ISS   
#wait
#python3 create_mass2dhist_V2.py --filename trees/ISS/Be9ISS_RICHNaF_V1_0.root --treename amstreea_cor --isotope Be9 --detector NaF

#counts be10 NaF
#wait
#python3 write_tree_q.py --filename trees/ISS/BeISS_BetaCor.root --treename amstreea_cor  --nuclei Be --isotope Be10 --qsel 4.0 --isIss 1  --outputprefix Be10ISS_RICHNaF_V1 --resultdir trees/ISS --detector NaF --betatype CIEMAT --datatype ISS
#wait
#python3 create_mass2dhist_V2.py --filename trees/ISS/Be10ISS_RICHNaF_V1_0.root --treename amstreea_cor --isotope Be10 --detector NaF
#wait
#root /home/manbing/Documents/LithiumAnalysis_RWTH/Scripts/Be_Analysis_V2/combinedFit_Sys_main_Be7.C
#root /home/manbing/Documents/LithiumAnalysis_RWTH/Scripts/Be_Analysis_V2/combinedFit_Sys_main_Be9.C 
#root /home/manbing/Documents/LithiumAnalysis_RWTH/Scripts/Be_Analysis_V2/combinedFit_Sys_main_Be10.C 

#acceptance
#wait
#python3 calculate_acceptance.py --variable "KineticEnergyPerNucleon"
#wait
#rich efficiency
#python3 calculate_rich_efficiency_correction.py --treename amstreea_cor
#python3 calculate_rich_efficiency_correction_naf.py --treename amstreea_cor
#python3 calculate_rich_efficiency_correction_agl.py --treename amstreea_cor

#python3 calculate_beflux.py


