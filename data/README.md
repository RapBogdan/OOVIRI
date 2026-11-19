# Date

Folderul `data/` e ignorat de Git, în afară de acest fișier și de `data/photos/` (fotografiile proprii, măsurătorile și plasările lor, vezi [photos/README.md](photos/README.md)).

```
data/
  nyu_test/<id>.png             imaginea NYU (640x480), descărcată cu scripts/download_nyu.py
  nyu_test/<id>_depth.png       adâncimea reală, uint16, în milimetri
  raw/                          fișierele parquet descărcate
  depth/<id>_<model>.npy        adâncimea estimată, float32, metri, la rezoluția imaginii
  photos/<nume_poza>.jpg        fotografii proprii, urcate în Git
  photos/masuratori.json        puncte de control măsurate cu ruleta
  photos/plasari.json           cazurile rulate: unde e pus obiectul virtual în fiecare poză
```

## Ce trece de la P1 la P2

Pentru fiecare imagine și fiecare model, o hartă de adâncime:

- `float32`, în metri, aliniată;
- la rezoluția imaginii; altă rezoluție cu același raport merge, se redimensionează la citire;
- numită `<id>_<model>.npy`, cum scrie `depth.run`; `<model>` este `depth_anything_v2`, `midas`, `zoedepth` sau `ruta_C`.

Hărțile numite `ruta_*` nu votează: primesc mască proprie și apar în panouri lângă rutele A, B și B-soft.

Adâncimea reală se citește fie din `.npy` (metri), fie din PNG pe 16 biți (milimetri), cum o scrie `scripts/download_nyu.py`.

## Ponderile modelelor

`config/priors.json` conține, pentru fiecare model, ponderea votului și eroarea lui tipică:

```json
{
  "depth_anything_v2": {"weight": 0.893, "abs_rel": 0.105},
  "midas":             {"weight": 0.853, "abs_rel": 0.122},
  "zoedepth":          {"weight": 0.936, "abs_rel": 0.084}
}
```

- `weight` = acuratețea modelului; ponderea votului lui în rutele B și B-soft.
- `abs_rel` = eroarea relativă medie de adâncime; spune cât de sigur e un vot în ruta B-soft.

Valorile de acum sunt provizorii: δ1 și AbsRel din evaluarea pe NYU, cu aliniere pe ZoeDepth.
