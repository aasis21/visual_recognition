# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import hashlib
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

md5('./models/course_model2.h5')


# course_model2.h5 : 3153daf7bb12a61dfed3b7484d4b1e52
# flower : fine_flowers2.h5 : 03348dd258b90501968af4417238b8f1
# aircrafts : fine_aircrafts.h5 : a04eed7aedb5dd19828a3dccc593a991
# dogs : fine_dogs.h5 : 3de8ce488064b3bcb776a8a138dd5369
# cars : fine_cars_best.h5 : 86c6d9849410b2d032e36d740107ca2a
# birds :



################33 NEW

# course_model2.h5 : 3153daf7bb12a61dfed3b7484d4b1e52

# flower : fine_flowers_temp100 : 9962d78df61d00660b5168eb189fc786

# aircrafts : fine_aircrafts_temp94.h5.h5 : b5b68d886cdedc5eea6e8ac1c566ef86

# dogs : fine_dogs_temp96.h5 : f8bd28efe18a7717236e8ca3230b12d0

# cars : fine_cars_temp80.h5.h5 : 43da6649f1d7227c323abed6e73fe3d2

# birds :fine_birds_72.h5: 8838ea04db35017a6666a7530ef9f130

Coarse : 3153daf7bb12a61dfed3b7484d4b1e52
fine: 
    flowers : 9962d78df61d00660b5168eb189fc786
    aircrafts : b5b68d886cdedc5eea6e8ac1c566ef86
    dogs : f8bd28efe18a7717236e8ca3230b12d0
    cars : 43da6649f1d7227c323abed6e73fe3d2
    birds : 8838ea04db35017a6666a7530ef9f130