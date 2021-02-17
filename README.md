# HPC_multiThreading
mpicc mandel_para.c -o mandel_para -lm && mpirun --n 12 --oversubscribe ./mandel_para 3840 3840 0.35 0.355 0.353 0.358 200