python ../join.py /home/test/DynamicFold/data/assembly/assembly.csv /home/test/DynamicFold/models/hybrid/H15/2/outputs.csv ../joined.csv
python cleavage_mae.py ../joined.csv cleavage_mae.png
python density_mae.py ../joined.csv density_mae.png
python depth_mae.py ../joined.csv depth_mae.png
python length_mae.py ../joined.csv length_mae.png
python mismatch_mae.py ../joined.csv mismatch_mae.png
python mae_distribution.py ../joined.csv mae_distribution.png
python prediction_reactivity.py ../joined.csv 100000 prediction_reactivity.png
