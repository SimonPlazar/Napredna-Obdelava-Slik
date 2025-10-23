# Odstranjevanje šuma

Naloga obsega

Generiranje umetnih učnih podatkov 
Trikotniki, štirikotniki, elipse, zvezde (s črtami) 
Naključne barve
Dodajanje šuma (gausov aditivni šum in pegast multiplikativen šum) 
Izdelavo in učenje nevronske mreže za odstranjevanje šuma 
Odstranjevanje aditivnega šuma 
Odstranjevanje multiplikativnega šuma 
Ovrednotenje naučene mreže
Rezultati na generiranih podatkih 
Delovanje na resničnih fotografijah
Generiranje učnih slik

Pripravimo prazno sliko in jo pobarvamo v naključno barvo
Učne slike naj bodo velike 256 x 256 pikslov
V sliko dodajamo/rišemo naključne like: trikotnike, štirikotnike, elipse, zvezde. Vsak lik pobarvamo z drugo naključno barvo.
Trikotnike in štirikotnike lahko generiramo tako, da izberemo 3 ali 4 naključne točke in izrišemo poligon. Ti bodo v slike dodali ostre robove in enobarvne površine.
Za elipso je najenostavneje izbrati naključno središče v sliki, naključen kot rotacije ter dva naključna polmera. Te bodo v sliko dodale nekaj krivulj.
Za zvezdo izberemo eno točko kot središče in 4 ali 5 točk kot krake zvezde. Vsak krak povežemo s središčem.
Za dodajanje šuma implementirajte dva tipa: aditivni in multiplikativni šum
Za simuliranje aditivnega šuma uporabite normalno porazdeljene vrednosti s povprečjem 0 in standardnim odklonom naključno izbranim v območju (za vsako sliko) med 0.1 in 0.3.
Za simuliranje multiplikativnega šuma uporabite normalno porazdeljene vrednosti s povprečjem 1 in standardnim odklonom naključno izbranim v območju (za vsako sliko) med 0.1 in 0.3.
Za aditivni šum vrednosti prištejete v sliko, za multiplikativni pa vrednosti s sliko pomnožite.
Preden sliko podate v mrežo, se morate prepričati, da so njene vrednosti še vedno smiselne.
Za risanje je smiselno uporabiti knjižnic, ki omogoča risanje z "anti aliasing". Kot primer lahko uporabite knjižnico OpenCV in funkcije fillPoly, ellipse, line.

Za učenje v vsako sliko narišite 5 primerkov vsake vrste lika (trikotnik, štirikotnik, elipsa, zvezda).

V poročilu pokažite nekaj primerov generiranih slik, kjer bodo prikazani vsi liki.

Nevronska mreža

Za nevronsko mrežo pripravite arhitekturo, ki se bo zgledovala po metodah filtriranja z ohranjenjem robov.

V slikah načeloma želimo ohraniti ostre robove. To lahko storimo tako, da za vsak piksel pogledamo, kakšna je orientacija roba v njegovi okolici.

Če je v njegovi bližini oster rob, potem piksel zgladimo s filtrom, ki je poravnan vzporedno s tem robom. Če v bližini piksla ni nobenega roba, ga lahko zgladimo z okroglim filtrom.

Vaša mreža naj bo tako sestavljena iz dveh vej:

V prvi veji boste celotno sliko filtrirali z različnimi filtri - smeri in oblike naj se mreža nauči sama.
V drugi veji boste za posamezen piksel ocenili razrede, ki bomo povedali, katerega izmed filtrov prve veje boste za posamezen piksel uporabili.
Na koncu veji združite tako, da razrede druge veje uporabite kot uteži za kombiniranje filtriranja prve veje.

Prva veja mreže:

razreže sliko na posamezne barvne kanale (R, G, B)
vsak barvni kanal ločeno filtrira z 8 konvolucijskimi filtri
velikost konv. filtrov 11x11, linearna prenosna funkcija
pazite, da ne pokvarite velikosti slike
Druga veja mreže:

sliko obdela s tremi ResNet bloki
32 kanalov, 3x3 filtri
"drop out" ali "batch normalization" sloj
ReLU prenosna funkcija
"residual" povezava
z dodatno konvolucijo pretvori 32 kanalov v 8 razredov
dodaj softmax sloj nad 8 kanalov, da izbere enega
Za združitev obeh vej vzamete izhode prve veje za posamezen barvni kanal, jih pomnožite z izhodom druge veje in seštejete preko kanalov.

To storite za vsak barvni kanal posebej, dobili bomo 3 slike z enim kanalom. Te konkateniramo skupaj, da dobimo novo RGB sliko. To je filtriran izhod mreže.

Nekatere možne konfiguracije:

Funkcije izgube: MSE (mean square error), RMSE (root mean square error), MAE (mean absolute error)
Adam/AdamW optimizator (lr ~ 1e-3)
Dobre rezultate bi morali doseči po 100000 korakih/vzorcih (1000 korakov je primerno za preizkušanje)
Evalvacija in demonstracija

Za ovrednotenje mreže uporabite metriko SNR (signal to noise ratio) in PSNR (peak signal to noise ratio).

Kvaliteto naučene mreže ocenite na 1000 generiranih slikah.

Kvaliteto ocenite tudi na nekaj izbranih realnih fotografijah

Ker zanje ne boste imeli slike brez šuma, bo ta ocena kvalitativna.
V poročilu prikažite te slike in izpostavite podrobnosti, kjer je mreža ohranila ali odstranila pomembne podrobnosti fotografije.
V poročilu pokažite tudi nabor filtrov, ki se jih je mreža naučila (ločeno za adidivni in multiplikativni šum).