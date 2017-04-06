iter=300
gcc train_hmm.cpp -o train_hmm -O3
time ./train_hmm $iter ../model_init.txt ../seq_model_01.txt model_01.txt &
time ./train_hmm $iter ../model_init.txt ../seq_model_02.txt model_02.txt &
time ./train_hmm $iter ../model_init.txt ../seq_model_03.txt model_03.txt &
time ./train_hmm $iter ../model_init.txt ../seq_model_04.txt model_04.txt &
time ./train_hmm $iter ../model_init.txt ../seq_model_05.txt model_05.txt &

gcc test_hmm.cpp -o test_hmm
./test_hmm