# Super ločljivost

Metode super ločljivosti, ki uporabljajo nevronske mreže, se v grobem delijo v dve skupini.

V prvo spadajo tiste, ki opravijo povečanje slike nizke ločljivosti v sami nevronski mreži. V tem primeru je izhodna slika večja od vhodne slike za izbran faktor povečanja.

Druge, bolj preproste, so tiste, ki sliko nizke ločljivosti najprej povečajo za izbran faktor in nato povečano sliko uporabijo
na vhodu nevronske mreže. Velikost slike na izhodu nevronske mreže je tako enaka velikosti slike na njenem vhodu.

V tej nalogi boste implementirali eno izmed metod, ki spada v drugo skupino. Podroben opis metode najdete v naslednjem članku.

Funkcionalnosti rešitve

Priprava učnih in testnih podatkov
pretvorba originalne slike v sliko nizke ločljivosti
pretvorba slike iz barvnega prostora RGB v YCbCr
ustvarjanje učnih slik s pomočjo okna
Gradnja nevronske mreže
pravilna struktura nevronske mreže
izpis strukture nevronske mreže
Učenje nevronske mreže - obvezno paketno učenje
izbira ustreznih parametrov učenja
učenje s faktorjem povečanja 2
učenje s faktorjem povečanja 3
Testiranje naučene nevronske mreže
prikaz naučenih filtrov prve konvolucijske plasti - izris filtriranih slik ne filtrov
ovrednotenje s pomočjo metrik PSNR in SSIM
prikaz delovanja na Set5 in Set14 zbirkah s podrobnejšim prikazom izbranega odseka na fotografiji
Priprava učnih in testnih podatkov

Za učenje in testiranje nevronske mreže bomo uporabili že pripravljene podatkovne zbirke, ki vsebujejo fotografije različnih motivov.

Za učenje lahko uporabite eno izmed podatkovnih zbirk: T91, BSDS300 ali DIV2K. Za testiranje uporabite podatkovni zbirki Set5 in Set14.

Za podatkovne zbirke, ki jih boste uporabili izpišite: število slik, povprečna širina slik, povprečna višina slik.

Slik iz izbranih podatkovnih zbirk ne uporabimo neposredno za učenje in testiranje nevronske mreže, temveč opravimo še 2 oziroma 3 korake predobdelave.

V prvem koraku predobdelave originalne slike pretvorimo v slike z nizko ločljivostjo. To storimo tako, da originalno sliko najprej pomanjšamo za nek izbran faktor ter jo nato povečamo na začetno velikost. Za pomanjšanje in povečanje slike uporabimo bilinearno interpolacijo. Slika nizke ločljivosti predstavlja vhod v nevronsko mrežo, medtem ko na izhodu pričakujemo originalno sliko. V tej nalogi kot faktor povečanja uporabite vrednosti 2 in 3.

V drugem koraku predobdelave slike pretvorimo iz barvnega prostora RGB v barvni prostor YCbCr. Na vhodu v nevronsko mrežo in na njenem izhodu uporabimo samo kanal Y. Na ta način zmanjšamo kompleknost vhodnih podatkov in nevronske mreže, pri tem pa rezultatov ne poslabšamo veliko, saj se večina za ta problem pomembnih informacij nahaja ravno v kanalu Y. Pri rekonstrukciji celotne slike za kanala Cb in Cr uporabimo njune interpolirane vrednosti iz slike nizke ločljivosti.

Tretji korak predobdelave uporabimo samo na učni podatkovni zbirki. V tem koraku iz originalnih slik z oknom velikosti 33x33 na naključnih mestih izrežemo sličice. Učna podatkovna zbirka bo po tem koraku sestavljena iz več sličic velikosti 33x33. Tega koraka na testnih podatkovnih zbirkah ne uporabimo, temveč med testiranjem na vhod nevronske mreže podamo sliko originalne velikosti. Učnih sličic velikosti 33x33 naj bo vsaj 1000.

Možen postopek priprave podatkov:

naložimo RGB sliko [original]
pretvorimo v RGB sliko z nizko ložljivostjo [low-res]
[original] sliko in [low-res] sliko pretvorimo v YCbCr
v tej točki imamo dva polja slik - v enem [original-ycbcr] slike ter v drugem [low-res-ycbcr] slike
iz vsake slike moramo izrezat več manjših sličic (za T91 bi izrezali 11 sličic iz vsake slike)
možen pristop
vsako sliko iz prejšnjih korakov v polje shranimo 11-krat
polja slik velikost 91 se razširijo na polja velikost 1001
sličice velikost 33x33 na naključnem mestu izrezujemo v DataLoader-ju
s tem dosežemo, da bodo uporabljene vedno druge naključne sličice
sličico izrežemo na enakem mestu v [original-ycbcr] in v [low-res-ycbcr] - dobimo vhodno-izhodne pare
Gradnja nevronske mreže

Nevronska mreža naj ima naslednjo strukturo:

Conv2D (64 filtrov velikosti 9x9)
ReLU
Conv2D (32 filtrov velikosti 5x5)
ReLU
Conv2D (1 filter velikosti 5x5)
Pri posamezni konvoluciji pazite, da bo širina in višina izhodnega tenzorja enaka širini in višini vhodnega tenzorja, saj želimo slike nespremenjenih velikosti.

To najlažje dosežete z nastavitvijo parametra "padding" na ustrezno vrednost.

Učenje nevronske mreže

Za učenje nevronske mreže lahko uporabite naslednje parametre ali pa njihove vrednosti postavite na vam smiselne:

Optimizacijski algoritem: Adam/AdamW optimizator (lr ~ 1e-5)
Funkcije izgube: MSE (mean square error), RMSE (root mean square error), MAE (mean absolute error)
Število epoh: 5000
Velikost batch: 32
Velikost okna: 33x33
Dobre rezultate bi morali doseči že med nekje 2000 in 3000 epohami, vendar boste z višjim številom epoh dosegli boljše rezultate.

Mrežo učite posebej za faktor povečanja 2 ter faktor povečanja 3.

V poročilo zapišite vse uporabljene parametre učenja.

Testiranje naučene nevronske mreže

Testiranje naučene nevronske mreže izvedite na testnih podatkovnih zbirkah Set5 in Set14.

Delovanje mreže ovrednotite s pomočjo metrik PSNR (Peak signal-to-noise ratio) in SSIM (Structural similarity index measure).

Za izračun uporabite samo vrednosti iz kanala Y. Uporabite lahko na primer funkciji skimage.metrics.peak_signal_noise_ratio in skimage.metrics.structural_similarity.

Poročajte povrečno vrednost in standardni odklon za posamezno podatkovno zbirko med originalno sliko in sliko nizke ločljivosti ter med originalno sliko in povečano sliko visoke ločljivosti.

Delovanje metode bolj podrobno prikažite tudi na vseh fotografijah iz omenjenih dveh testnih podatkovnih zbirk. Za posamezno sliko prikažite njen original, sliko nizke ločljivosti ter sliko visoke ločljivosti. Nad posamezno sliko zapišite izračunani metriki PSNR in SSIM. Nato si izberite nek manjši odsek na fotografiji, kjer ocenite, da je delovanje metode dobro vidno, in tisti del posebej prikažite.

Nad slikami želimo videt vrednosti PSNR in SSIM v kombinacijah Original proti LowRes in Original proti Predicted (HighResPredicted).

V poročilu izrišite tudi nabor slik po aplikaciji naučenih filtrov v prvem konvolucijskem sloju.

Oddaja naloge

Poročilo in program naj bo izdelan v Jupyther Notebook (.ipynb), zaledne funkcije lahko implementirate v ločenih .py datotekah. Na sistem oddajte datoteko naloga.zip, ki naj vsebuje porocilo.ipynb datoteko in njeno .html različico (izvoz). Prav tako oddajte vse lastne datoteke, ki se nanašajo na delovanje programa (.py datoteke). Ostalih datotek ne oddajajte! Poročilo naj v celicah vsebuje morebitne odgovore na zastavljena vprašanja, vse potrebne grafe ali slike ter ustrezno komentirane poglavitne dele kode programa. V primeru nespoštovanja predpisane oblike poročanja bo naloga zavrnjena in ocenjena z 0 točkami.


