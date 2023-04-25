#! /usr/bin/bash

for (( n = 1; n < 5; n++))
do
  echo "n = $n"
  for (( i = 0; i < 100; i++))
  do
    mpirun -np $n ./lab1
  done
done
