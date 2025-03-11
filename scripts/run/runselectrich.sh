python3 write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be7MC_NucleiSelection.root --qsel 4.0 --outputprefix Be7MC_NucleiSelection_RICHNaF --resultdir /home/manbing/Documents/Data/data_mc
  --betatype CIEMAT --detector NaF --nuclei Be --isotope Be7 --qsel 4.0  --datatype MC
wait
python3 write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be9MC_NucleiSelection.root --qsel 4.0 --outputprefix Be9MC_NucleiSelection_RICHNaF --resultdir /home/manbing/Documents/Data/data_mc
 --betatype CIEMAT --detector NaF --nuclei Be --isotope Be7 --qsel 4.0 --datatype MC    
wait
python3 write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be10MC_NucleiSelection.root --qsel 4.0 --outputprefix Be10MC_NucleiSelection_RICHNaF --resultdir /home/manbing/Documents/Data/data_mc --betatype CIEMAT --detector NaF --nuclei Be --isotope Be7 --qsel 4.0 --datatype MC    
wait

python3 write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be7MC_NucleiSelection.root --qsel 4.0 --outputprefix Be7MC_NucleiSelection_RICHAgl --resultdir /home/manbing/Documents/Data/data_mc
  --betatype CIEMAT --detector Agl --nuclei Be --isotope Be7 --qsel 4.0 --datatype MC    
wait
python3 write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be9MC_NucleiSelection.root --qsel 4.0 --outputprefix Be9MC_NucleiSelection_RICHAgl --resultdir /home/manbing/Documents/Data/data_mc
 --betatype CIEMAT --detector Agl --nuclei Be --isotope Be7 --qsel 4.0 --datatype MC    
wait
python3 write_tree_q.py --filename /home/manbing/Documents/Data/data_mc/Be10MC_NucleiSelection.root --qsel 4.0 --outputprefix Be10MC_NucleiSelection_RICHAgl --resultdir /home/manbing/Documents/Data/data_mc --betatype CIEMAT --detector Agl --nuclei Be --isotope Be7 --qsel 4.0 --datatype MC    

#python3 calculate_acceptance.py --treename amstreea --variable Rigidity
