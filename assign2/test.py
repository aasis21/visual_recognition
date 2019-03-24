# -*- coding: utf-8 -*-

act_ = open("./output_n.txt", "r")
act = act_.read()
act = act.split("\n")
actual = {}
for each in act[:-1]:
    name = each.split(" ")
    actual[name[0]] = name

pred_ = open("./output.txt", "r")
pred = pred_.read()
pred = pred.split("\n")

fine_true = 0
crs_true = 0
total = len(actual)

for each in pred[:-1]:
    name = each.split(" ")
    ac =  actual[name[0]]
    if(ac[1] == name[1] ):
        print(ac, name)
        crs_true = crs_true + 1
        if(int(ac[2].split('@')[1]) == int(name[2].split('@')[1])+1):
            fine_true = fine_true + 1
        
print(crs_true)
print(fine_true)
print(total)
print(crs_true/total)
print(fine_true/total)
    
