vmtouch -vt metrics.db -vt rtstops.db
python format.py SRRList.csv metrics.db rtstops.db genome/GRCh38_lt.fa depths.csv 0.25 False neural.db neural 16
python references.py genome/annotations.csv neural.db
md5sum neural.db > neural.md5
chmod 444 neural.db neural.md5
