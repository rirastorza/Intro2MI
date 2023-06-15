//+
SetFactory("OpenCASCADE");
sx = 14.0;//longitud de la caja en x
sy = 14.0;//longitud de la caja en y
r = 1.0;//radio del cilindro
Rectangle(1) = {-sx/2, -sy/2, 0, sx, sy, 0};
Disk(2) = {0, 0, 0, r, r};
BooleanDifference(3) = { Surface{1};Delete;}{Surface{2};};


Physical Surface(1) = {3};//medio de acoplamiento
Physical Surface(2) = {2};//cilindro

Physical Line(10) = {6,7,8,9};//cilindro

Mesh.CharacteristicLengthMax = 0.2;
