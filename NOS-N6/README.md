# Doslikava

V tej nalogi boste implementirali prirejeno metodo doslikave, ki se zgleduje po arhitekturi U-Net in delne konvolucije. Delovanje metode boste preverili na umetno generiranih slikah z besedilom.

V nalogi bomo zasledovali članek Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, Bryan Catanzaro: Image Inpainting for Irregular Holes Using Partial Convolutions

Glede na to, da smo se z U-Net arhitekturo že spoznali, bomo zaradi optimizacije časa učenja arhitekturo iz članka močno reducirali ter uporabili dosti manjši model, ki ne bo popoln U-Net.

To bo seveda povročilo slabše rezultate doslikavanja. Lahko (popolnoma neobvezno) prilagodite U-Net model iz naloge Semantična segmentacija (vse konvolucijske sloje Conv2d zamenjate za delno konvolucijo PartialConv2d).



Funkcionalnosti rešitve

generiranje umetnih učnih in testnih slik
uporaba naključne fotografije
izris črt (zatemnitev delov) na sliko - generiranje maske
gradnja nevronske mreže
pravilna in delujoča struktura nevronske mreže
učenje nevronske mreže
paketno učenje
spremljanje napake in natančnosti
testiranje naučene nevronske mreže
testiranje nevronske mreže z umetnimi slikami
ovrednotenje s pomočjo lastno izbranih metrik
Priprava učnih in testnih podatkov

Za učenje in testiranje nevronske mreže uporabite umetno generirane slike. To vam bo omogočalo učenje in testiranje na poljubnem številu slik.

Generirali boste torej sliko, masko in maskirano sliko (sliko, ki ima manjkajoče dele). Velikost posamezne slike naj bo 256 x 256 pikslov.

V nadaljevanju uporabite podatkovno zbirko DIV2K (vključuje učno in validacijsko zbirko slik).

Iz naključno izbrane fotografije na naključnem mestu izrežite sliko velikosti 256 x 256 pikslov, ki vam predstavlja sliko, katero kasneje maskirate.

Za vsako sliko generirajte nakjlučno masko, na katero zaršite naključne debelejše črte (kot v primeru poročila).

Maska je enodimenzionalna slika, z vrednostmi 1, razen na mestih, kjer je zarisana črta - tam so vrednosti 0. Vrednosti 0 torej predstvaljajo luknjo v sliki - manjkajoči del slike.

Ko masko apliciramo na sliko, bo slika, kjer ima maska vrednosti 0 postala črna.

Pred samim učenjem vrednosti vhodnih slik ustrezno skalirajte med 0 in 1, ter preuredite vrstni red dimenzij, da bo ustrezal tistemu, ki ga pričakuje ogrodje za učenje.

V poročilo zapišite naslednje lastnosti generiranih slik:

širina slik
višina slik
število učnih slik
število testnih slik
V poročilu prav tako izrišite nekaj umetno generiranih slik (izvorna slika, maskirana slika in maska).

Gradnja nevronske mreže

Struktura nevronske mreže se zgleduje po arhitekturi U-Net z delno konvolucijo a je močno reducirana, kar omogoča hitrejše učenje, so pa rezultati lahko slabši kot v referenčnem članku:

PartialBlock (32, 3x3, str=1) → PartialBlock (64, 3x3, str=2) → PartialBlock (128, 3x3, str=2)
PartialBlock (128, 3x3, str=2) → UpSample (scl_fact=2)
PartialBlock (64, 3x3, str=1) → UpSample (scl_fact=2) → PartialBlock (32, 3x3, str=1) → UpSample (scl_fact=2) → PartialConv2d (3, 3x3, str=1)
PartialBlock ima naslednjo strukturo:

PartialConv2d (str=1 ali str=2) → BatchNorm2d → ReLU
PartialConv2d ima naslednjo implementacijo:

Prejme parametre (x, maska)
Na x apliciramo masko
Na x apliciramo klasično konvolucijo (Conv2d(3x3, stride, 1))
Na masko apliciramo konvolucijo z operatorjem konvolucije (3x3) s samimi enicami
Na ta način "preštejemo" koliko vrednosti je "veljavnih" (ima vsaj eno vrednost pod operatorjem enoko 1)
Na podlagi dobljenega rezultata določimo novo masko - maska se krči po robovih
Izvedemo normalizacijo, saj klasična konvolucija predpostvalja uporabo vseh vrednosti
x = x * (self.conv_all_ones.numel()/(mask_sum + 1e-8)) * mask_new
Vrnemo x in novo masko


Nevronska mreža ima poleg običajnih zaporednih povezav naprej tudi dodatne povezave, ki povezujejo PartialBlock z enakim številom konvolucijskih filtrov. V splošnem se PartialBlock z indeksom i na strani enkoderja poveže z PartialBlock z indeksom i na strani dekoderja.

Pri posamezni konvoluciji pazite, da bo širina in višina izhodnega tenzorja enaka širini in višini vhodnega tenzorja, saj želimo slike nespremenjenih velikosti. To najlažje dosežete z nastavitvijo parametra “padding” na ustrezno vrednost.

Vizualna predstavitev modificirane arhitekture nevronske mreže:



V poročilu prikažite strukturo zgrajene nevronske mreže. V pomoč vam je lahko na primer funkcija torchinfo.summary.

Učenje nevronske mreže

Za učenje nevronske mreže lahko uporabite naslednje parametre ali pa njihove vrednosti postavite na vam smiselne:

Optimizacijski algoritem: Adam
Hitrost učenja: 0.0001
Metrika izgube: L1
Število epoh: min. 1000
Velikost batch: 32
Širina slik: 256
Višina slik: 256
Število slik na epoho: min. 1024
# Funkcija izgube
def loss_fn(predicted, image, mask):
hole = 1 - mask
return ((predicted-image).abs() * hole).sum() / (hole.sum() + 1e-8)
Dobre rezultate bi morali doseči že pri nekje 50 epohah (pri 1024 slikah na epoho), vendar boste z višjim številom epoh dosegli boljše rezultate.

V poročilo zapišite vse uporabljene parametre učenja.

Izrišite graf napake učenja (loss).

Testiranje naučene nevronske mreže

Naučeno nevronsko mrežo vizualno testirajte na 10 umetno generiranih slikah.

Izrišite štiri slike: izvorna slika, maskirana slika, maska in izhod in NM.

Dodatno nevronsko mrežo statistično testirajte na 1000 umetno generiranih slikah.

Delovanje mreže ovrednotite s pomočjo po lastni presoji izbranih metrik.

Poročajte povrečno vrednost in standardni odklon za posamezno metriko.

Oddaja naloge

Poročilo in program naj bo izdelan v Jupyther Notebook (.ipynb), zaledne funkcije lahko implementirate v ločenih .py datotekah. Na sistem oddajte datoteko naloga.zip, ki naj vsebuje porocilo.ipynb datoteko in njeno .html različico (izvoz). Prav tako oddajte vse lastne datoteke, ki se nanašajo na delovanje programa (.py datoteke). Ostalih datotek ne oddajajte! Poročilo naj v celicah vsebuje morebitne odgovore na zastavljena vprašanja, vse potrebne grafe ali slike ter ustrezno komentirane poglavitne dele kode programa. V primeru nespoštovanja predpisane oblike poročanja bo naloga zavrnjena in ocenjena z 0 točkami.