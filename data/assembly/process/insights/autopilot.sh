python histogram.py ../assembly.db assembly "['MeanEnd', 'MeanMismatch']" "" 100 True 1e-5 original
python histogram.py ../assembly.db assembly "['Start', 'End', 'FullLength', 'ValidLength', 'StripLength', 'MeanDepth', 'MeanDensity', 'Gap']" "" 100 True  1 original
python histogram.py ../assembly.db assembly "['MeanEnd', 'MeanMismatch']" "WHERE FullLength BETWEEN 64 AND 4096 AND StripLength <= 16 AND MeanDepth >= 16 AND MeanDensity >= 64 AND Gap = 0" 100 True 1e-5 filtered
python histogram.py ../assembly.db assembly "['Start', 'End', 'FullLength', 'ValidLength', 'StripLength', 'MeanDepth', 'MeanDensity', 'Gap']" "WHERE FullLength BETWEEN 64 AND 4096 AND StripLength <= 16 AND MeanDepth >= 16 AND MeanDensity >= 64 AND Gap = 0" 100 True 1 filtered
python completeness.py ../assembly.db assembly "WHERE FullLength BETWEEN 64 AND 4096 AND StripLength <= 16 AND MeanDepth >= 16 AND MeanDensity >= 64 AND Gap = 0" 16 filtered 16
python information.py ../assembly.db assembly SeqID "['A', 'C', 'G', 'U', 'RD', 'ER', 'MR', 'IC', 'RT']" "WHERE FullLength BETWEEN 64 AND 4096 AND StripLength <= 16 AND MeanDepth >= 16 AND MeanDensity >= 64 AND Gap = 0" filtered
