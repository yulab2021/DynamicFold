vmtouch -vt /home/test/DynamicFold/data/neural/process/neural.db -vt /home/test/DynamicFold/data/zebrafish/process/zebrafish.db
python assembly.py /home/test/DynamicFold/data/neural/process/neural.db neural assembly.db assembly
python assembly.py /home/test/DynamicFold/data/zebrafish/process/zebrafish.db zebrafish assembly.db assembly
python assembly.py /home/test/DynamicFold/data/neural/process/neural.db ref assembly.db ref
python assembly.py /home/test/DynamicFold/data/zebrafish/process/zebrafish.db ref assembly.db ref

cp assembly.db ../assembly.db
md5sum assembly.db > assembly.md5
chmod 444 assembly.db assembly.md5

python query.py assembly.db debug.csv "SELECT COUNT(*) FROM assembly WHERE FullLength BETWEEN 64 AND 4096 AND StripLength <= 16 AND MeanDepth >= 0.1 AND MeanDensity >= 64 AND Gap = 0"
python query.py assembly.db ../sample.csv "SELECT * FROM assembly WHERE FullLength BETWEEN 64 AND 4096 AND StripLength <= 16 AND MeanDepth >= 0.1 AND MeanDensity >= 64 AND Gap = 0 ORDER BY RANDOM() LIMIT 10"
python query.py assembly.db ../assembly.csv "SELECT * FROM assembly WHERE FullLength BETWEEN 64 AND 4096 AND StripLength <= 16 AND MeanDepth >= 0.1 AND MeanDensity >= 64 AND Gap = 0 ORDER BY RANDOM()"

python insights/histogram.py assembly.db assembly "['MeanEnd', 'MeanMismatch']" "" 100 True 1 1e-6 insights
python insights/histogram.py assembly.db assembly "['Start', 'End', 'FullLength', 'ValidLength', 'StripLength', 'Gap', 'MeanDensity']" "" 100 True 1 1 insights
python insights/histogram.py assembly.db assembly "['MeanDepth']" "" 100 True 100 1 insights

python insights/completeness.py assembly.db assembly "WHERE FullLength BETWEEN 64 AND 4096 AND StripLength <= 16 AND MeanDepth >= 0.1 AND MeanDensity >= 64 AND Gap = 0" 16 insights 16
python insights/information.py assembly.db assembly SeqID "['A', 'C', 'G', 'U', 'RD', 'ER', 'MR', 'IC', 'RT']" "WHERE FullLength BETWEEN 64 AND 4096 AND StripLength <= 16 AND MeanDepth >= 0.1 AND MeanDensity >= 64 AND Gap = 0" insights

