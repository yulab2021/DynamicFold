python predict_structure.py /home/test/DynamicFold/results/performance/joined.csv structure.csv bpps.db 8
python delta_mae.py structure.csv delta_mae.png
python dynamicity_squares.py structure.csv 50 dynamicity_squares.png
python metrics_curves.py structure.csv bpps.db curves 4
python metrics_tables.py structure.csv 50 tables 4
