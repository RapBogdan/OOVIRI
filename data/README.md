# Date

Folderul `data/` e ignorat de Git, în afară de acest fișier și de `data/photos/` (fotografiile proprii și măsurătorile lor, vezi [photos/README.md](photos/README.md)). Structura la care se așteaptă codul P2:

```
data/
  rgb/<id>.png                  imaginea reală (NYU: 640x480)
  gt/<id>.npy                   adâncimea reală, float32, metri, 0 = fără măsurătoare (doar NYU)
  depth/<model>/<id>.npy        adâncimea estimată de un model, float32, metri, aliniată, HxW ca imaginea
  priors.json                   ponderile și erorile modelelor, măsurate pe subsetul de validare
  photos/<nume_poza>.jpg        fotografii proprii, urcate în Git
  photos/masuratori.json        puncte de control măsurate cu ruleta
```

`<model>` este numele folderului: `depth_anything_v2`, `midas_dpt`, `zoedepth`. Pentru fotografiile proprii, `<id>` este numele pozei, fără extensie.

Exemplu de `priors.json` (numerele sunt doar de formă, cele reale le măsoară P1):

```json
{
  "depth_anything_v2": {"weight": 0.93, "abs_rel": 0.05},
  "midas_dpt":         {"weight": 0.86, "abs_rel": 0.09},
  "zoedepth":          {"weight": 0.91, "abs_rel": 0.07}
}
```

- `weight` = acuratețea modelului pe validare; ponderea votului lui în rutele B și B-soft.
- `abs_rel` = eroarea relativă medie de adâncime; spune cât de sigur e un vot în ruta B-soft.
