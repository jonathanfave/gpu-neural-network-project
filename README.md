# gpu-neural-network-project

Utilisation des jeux de données sur : http://yann.lecun.com/exdb/mnist/

Pour compiler le code :
nvcc -pg -o main *.c *.cu
L'option -pg permet de générer un fichier de perf lors de l'execution

Pour executer le code : 
./main

Pour mesurer les performances CPU :
  * Compiler le code
  * Executer le code
  * gprof -Q -b main

Pour mesurer les performances GPU :
  * Compiler le code
  * sudo /usr/local/cuda/bin/nvprof main
