=======================================================================================
List of possible test dataset captures
=======================================================================================

Clear
2013-05-26 15:15
2013-05-27 09:45
2013-05-27 10:15
2013-05-27 10:30
2013-05-27 12:00
2013-07-26 13:15
2013-09-24 15:39
2013-09-24 17:09

Scattered
2013-05-12 13:00
2013-04-14 14:24
2013-05-26 11:45
2013-05-26 12:15
2013-05-26 12:30
2013-05-30 12:45
2013-05-30 12:00
2013-07-29 09:15
2013-07-29 10:00
2013-08-30 09:15
2013-09-26 10:40
2013-09-26 11:50
2013-09-26 15:10

Overcast
2013-04-14 11:36
2013-04-15 08:40
2013-04-15 08:16
2013-07-29 13:30

=======================================================================================
Datasets and commands used for results of SPIE2018
=======================================================================================

mix
2013-05-26 15:15 c
2013-05-27 10:15 c
2013-07-26 13:15 c
2013-09-24 15:39 c
2013-05-12 13:00 s
2013-05-26 12:30 s
2013-07-29 10:00 s
2013-09-26 11:50 s
2013-04-14 11:36 o
2013-04-15 08:40 o
2013-07-29 13:30 o
python.exe .\run.py etr mix train -n 4 -t 50 -d 25 -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39, 2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50, 2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py etr mix test -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39, 2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50, 2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py rfr mix train -n 4 -d 30 -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39, 2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50, 2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py rfr mix test -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39, 2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50, 2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py knr mix train -n 4 -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39, 2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50, 2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py knr mix test -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39, 2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50, 2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py lnr mix train -n 4 -y 4 -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39, 2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50, 2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py lnr mix test -y 4 -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39, 2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50, 2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"

clear
2013-05-26 15:15 c
2013-05-27 10:15 c
2013-07-26 13:15 c
2013-09-24 15:39 c
python.exe .\run.py etr clear train -n 4 -t 50 -d 25 -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39"
python.exe .\run.py etr clear test -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39"
python.exe .\run.py rfr clear train -n 4 -d 30 -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39"
python.exe .\run.py rfr clear test -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39"
python.exe .\run.py knr clear train -n 4 -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39"
python.exe .\run.py knr clear test -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39"
python.exe .\run.py lnr clear train -n 4 -y 3 -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39"
python.exe .\run.py lnr clear test -y 3 -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39"

python.exe .\run.py etr clear plot -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39"
python.exe .\run.py rfr clear plot -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39"
python.exe .\run.py knr clear plot -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39"
python.exe .\run.py lnr clear plot -y 3 -c "2013-05-26 15:15, 2013-05-27 10:15, 2013-07-26 13:15, 2013-09-24 15:39"

scattered
2013-05-12 13:00 s
2013-05-26 12:30 s
2013-07-29 10:00 s
2013-09-26 11:50 s
python.exe .\run.py etr scattered train -n 4 -t 50 -d 25 -c "2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50"
python.exe .\run.py etr scattered test -c "2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50"
python.exe .\run.py rfr scattered train -n 4 -d 30 -c "2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50"
python.exe .\run.py rfr scattered test -c "2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50"
python.exe .\run.py knr scattered train -n 4 -c "2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50"
python.exe .\run.py knr scattered test -c "2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50"
python.exe .\run.py lnr scattered train -n 4 -y 3 -c "2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50"
python.exe .\run.py lnr scattered test -y 3 -c "2013-05-12 13:00, 2013-05-26 12:30, 2013-07-29 10:00, 2013-09-26 11:50"

overcast
2013-04-14 11:36 o
2013-04-15 08:40 o
2013-07-29 13:30 o
python.exe .\run.py etr overcast train -n 4 -t 50 -d 25 -c "2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py etr overcast test -c "2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py rfr overcast train -n 4 -d 30 -c "2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py rfr overcast test -c "2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py knr overcast train -n 4 -c "2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py knr overcast test -c "2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py lnr overcast train -n 4 -y 3 -c "2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"
python.exe .\run.py lnr overcast test -y 3 -c "2013-04-14 11:36, 2013-04-15 08:40, 2013-07-29 13:30"

=======================================================================================
Datasets and commands used for results of Solar Energy 2019 paper
=======================================================================================

python.exe .\sradmap.py -l -v -g -p "sandbox\etr\20130526_1515.tiff" -t "2013/05/26 15:15" -c 2 -m etr_clear-tiff-rgb.pkl
python.exe .\sradmap.py -l -v -g -p "sandbox\etr\20130527_1015.tiff" -t "2013/05/27 10:15" -c 2 -m etr_clear-tiff-rgb.pkl
python.exe .\sradmap.py -l -v -g -p "sandbox\etr\20130726_1315.tiff" -t "2013/07/26 13:15" -c 2 -m etr_clear-tiff-rgb.pkl
python.exe .\sradmap.py -l -v -g -p "sandbox\etr\20130924_1539.tiff" -t "2013/09/24 15:39" -c 2 -m etr_clear-tiff-rgb.pkl
python.exe .\sradmap.py -l -v -g -p "sandbox\rfr\20130526_1515.tiff" -t "2013/05/26 15:15" -c 2 -m rfr_clear-tiff-rgb.pkl
python.exe .\sradmap.py -l -v -g -p "sandbox\rfr\20130527_1015.tiff" -t "2013/05/27 10:15" -c 2 -m rfr_clear-tiff-rgb.pkl
python.exe .\sradmap.py -l -v -g -p "sandbox\rfr\20130726_1315.tiff" -t "2013/07/26 13:15" -c 2 -m rfr_clear-tiff-rgb.pkl
python.exe .\sradmap.py -l -v -g -p "sandbox\rfr\20130924_1539.tiff" -t "2013/09/24 15:39" -c 2 -m rfr_clear-tiff-rgb.pkl
python.exe .\sradmap.py -l -v -g -p "sandbox\knr\20130526_1515.tiff" -t "2013/05/26 15:15" -c 2 -m knr_clear-tiff-rgb.pkl -s knr_scaler_clear-tiff-rgb.pkl
python.exe .\sradmap.py -l -v -g -p "sandbox\knr\20130527_1015.tiff" -t "2013/05/27 10:15" -c 2 -m knr_clear-tiff-rgb.pkl -s knr_scaler_clear-tiff-rgb.pkl
python.exe .\sradmap.py -l -v -g -p "sandbox\knr\20130726_1315.tiff" -t "2013/07/26 13:15" -c 2 -m knr_clear-tiff-rgb.pkl -s knr_scaler_clear-tiff-rgb.pkl
python.exe .\sradmap.py -l -v -g -p "sandbox\knr\20130924_1539.tiff" -t "2013/09/24 15:39" -c 2 -m knr_clear-tiff-rgb.pkl -s knr_scaler_clear-tiff-rgb.pkl
python.exe .\sradmap.py -l -v -g -p "sandbox\lnr\20130526_1515.tiff" -t "2013/05/26 15:15" -c 2 -m lnr_clear-tiff-rgb.pkl -s lnr_scaler_clear-tiff-rgb.pkl -y 3
python.exe .\sradmap.py -l -v -g -p "sandbox\lnr\20130527_1015.tiff" -t "2013/05/27 10:15" -c 2 -m lnr_clear-tiff-rgb.pkl -s lnr_scaler_clear-tiff-rgb.pkl -y 3
python.exe .\sradmap.py -l -v -g -p "sandbox\lnr\20130726_1315.tiff" -t "2013/07/26 13:15" -c 2 -m lnr_clear-tiff-rgb.pkl -s lnr_scaler_clear-tiff-rgb.pkl -y 3
python.exe .\sradmap.py -l -v -g -p "sandbox\lnr\20130924_1539.tiff" -t "2013/09/24 15:39" -c 2 -m lnr_clear-tiff-rgb.pkl -s lnr_scaler_clear-tiff-rgb.pkl -y 3
