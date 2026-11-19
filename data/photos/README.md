# Fotografii proprii

Șase poze de interior făcute de P2, pe verticală, cu un Motorola Edge 60 (obiectivul principal la 1x, 24 mm echivalent). Sunt copiile trimise prin chat, deci nu au datele camerei în fișier; distanța focală vine din fișa tehnică a telefonului.

| Poză | Ce testează |
| --- | --- |
| `foto_01_scaun_lateral.jpg` | margini subțiri: picioarele scaunului |
| `foto_02_scaun_fata.jpg` | caz clar: scaun la 1,2-1,6 m, perdea la 2,3 m, loc pentru obiect între ele |
| `foto_03_rucsac.jpg` | obiect lângă faldurile unei perdele |
| `foto_04_monitor.jpg` | suprafață neagră lucioasă, cu peretele la 25 cm în spate |
| `foto_05_usa.jpg` | adâncime mare: tocul ușii la 1 m, ușa de vizavi la 2,6 m |
| `foto_06_oglinda.jpg` | caz greu: oglindă și geantă lucioasă |

## Pentru P1: ce hărți de adâncime trebuie

Pentru fiecare poză și fiecare model, câte un fișier:

```
data/depth/<model>/<nume_poza>.npy
```

- `float32`, în metri
- aliniată la scara ZoeDepth (modelele relative aduse la metri)
- aceeași rezoluție ca poza; altă rezoluție cu același raport merge, se redimensionează la citire
- `<model>` este numele folderului: `depth_anything_v2`, `midas_dpt`, `zoedepth`

Exemplu: `data/depth/zoedepth/foto_02_scaun_fata.npy`

## Puncte de control măsurate cu ruleta

Pozele proprii nu au adâncime reală, așa că la cinci dintre ele am măsurat distanța până la câteva puncte. Sunt în `masuratori.json`.

Ruleta măsoară pe podea, de sub telefon până sub punct. Harta de adâncime măsoară de la obiectiv, pe direcția în care privește camera. `ooviri/measure.py` transformă una în alta, folosind înălțimea la care a fost ținut telefonul și poziția punctului în poză.

```bash
.venv\Scripts\python -m ooviri check --photo foto_02_scaun_fata --depth zoedepth=data/depth/zoedepth/foto_02_scaun_fata.npy
```

Comanda afișează, pentru fiecare punct, adâncimea așteptată, cea din hartă și eroarea în procente. Fără `--depth` afișează doar adâncimile așteptate.

Verificare internă: la `foto_02_scaun_fata`, cele trei puncte de pe podea implică aceeași înclinare a telefonului (17,0, 17,6 și 17,1 grade), deci distanțele, înălțimea telefonului și distanța focală se potrivesc între ele.

La `foto_05_usa` gresia e pătrată, cu latura de 30 cm. Pasul dintre liniile de gresie, dedus doar din poză (înălțimea telefonului, înclinarea și distanța focală), iese 0,29-0,30 m. Trei dintre linii sunt folosite ca puncte de control: distanța până la ușa de vizavi minus un număr întreg de plăci.

La `foto_04_monitor` și `foto_06_oglinda` înălțimea telefonului nu e cunoscută, deci distanțele sunt folosite direct ca adâncime și sunt marcate ca aproximative.

## Calibrarea scării și plasările

Pe aceste poze hărțile de adâncime supraestimează distanțele cu 25% până la 85%. De aceea `ooviri batch --measurements masuratori.json` corectează fiecare model pe punctele măsurate ale pozei:

- cu cel puțin două puncte aflate clar la adâncimi diferite: scară și decalaj (`adevărat = a x hartă + b`);
- cu un singur punct, sau cu puncte la aproape aceeași adâncime: doar scară;
- fără puncte (`foto_01_scaun_lateral`): fără calibrare.

`plasari.json` conține cazurile rulate. Adâncimea obiectului nu e dată în metri, ci între două repere din poză (`z_intre`): de exemplu la jumătatea distanței dintre spătarul scaunului și perdea. Așa obiectul ajunge sigur în spatele unui lucru și în fața altuia, oricare ar fi scara hărții.
