from Solar_Wind_Tools.run_all import run_all

run_all('./test_omni.hdf5',
        key='omni_window_2h_20', window=2*60, shift=20, start_year= 2000, 
        end_year=2003, run=['Download', 'Statistics', 'Coupling', 'Dipole', 'Time Shift'])