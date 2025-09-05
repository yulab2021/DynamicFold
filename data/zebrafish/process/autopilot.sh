vmtouch -vt metrics.db -vt rtstops.db
python format.py SraList.csv metrics.db rtstops.db genome/GRCz11_lt.fa depths.csv 0.25 False zebrafish.db zebrafish 16
python references.py genome/fetch.csv zebrafish.db
md5sum zebrafish.db > zebrafish.md5
chmod 444 zebrafish.db zebrafish.md5
