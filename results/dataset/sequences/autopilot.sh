python base_type.py /home/test/DynamicFold/data/assembly/assembly.csv base_type.png
python motif_coverage.py /home/test/DynamicFold/data/assembly/assembly.csv 8 16 motif_coverage.png 8
python ../concat.py /home/test/DynamicFold/data/neural/process/genome/annotations.csv /home/test/DynamicFold/data/zebrafish/process/genome/fetch.csv ../annotations.csv
python biotype.py /home/test/DynamicFold/data/assembly/assembly.csv ../annotations.csv 6 1.65 biotype.png