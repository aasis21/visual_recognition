#!/bin/bash
array=($(ls test))
echo $array
for i in "${array[@]}"
do
   name=${i::-3}
   txt="txt"
   echo $name$txt
   python3 src/predict.py ./test/$i ./test_outputs/$name$txt
   sleep 2
   # or do whatever with individual element of the array
done
