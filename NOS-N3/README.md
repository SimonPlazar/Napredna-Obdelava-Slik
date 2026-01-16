# Semantična segmentacija

V tej nalogi boste implementirali rahlo prirejeno metodo semantične segmentacije, ki se zgleduje po arhitekturi U-Net. Delovanje metode boste preverili na umetno generiranih slikah z besedilom.

Podroben opis arhitekture U-Net najdete v naslednjem članku - Olaf Ronneberger, Philipp Fischer, Thomas Brox: U-Net: Convolutional Networks for Biomedical Image Segmentation.

Arhitektura U-Net nadgrajuje osnovno arhitekturo konvolucijske nevronske mreže za semantično segmentacijo, predstavljeno v naslednjem članku - Jonathan Long, Evan Shelhamer, Trevor Darrell: Fully Convolutional Networks for Semantic Segmentation.

Funkcionalnosti rešitve

generiranje umetnih učnih in testnih slik
uporaba naključne fotografije za ozadje
uporaba naključne fotografije za teksturo besedila
uporaba naključne velikosti in debeline črk ter tip pisave
augumentacija
gradnja nevronske mreže
pravilna in delujoča struktura nevronske mreže
učenje nevronske mreže
paketno učenje
izvajanje validacije med učenjem
spremljanje napake in natančnosti (učna in validacijska množica) - na koncu učenja graf
testiranje naučene nevronske mreže
testiranje nevronske mreže z umetnimi slikami
ovrednotenje s pomočjo metrik IoU in Dice koeficienta
prikaz delovanja na treh izbranih fotografijah z besedilom
Priprava učnih in testnih podatkov

Za učenje in testiranje nevronske mreže uporabite umetno generirane slike. To vam bo omogočalo učenje in testiranje na poljubnem številu slik.

Ker se vaja ukvarja s semantično segmentacijo besedila, želite generirati pare slik z besedilom in njihovo masko, ki za posamezen piksel pove ali predstavlja besedilo.

Slike z besedilom uporabite na vhodu nevronske mreže njihovo masko pa na izhodu. Velikost posamezne slike naj bo 256 x 256 pikslov.

V nadaljevanju uporabite podatkovno zbirko DIV2K (vključuje učno in validacijsko zbirko slik).

Iz naključno izbrane fotografije na naključnem mestu izrežite sliko velikosti 256 x 256 pikslov, ki vam predstavlja ozadje generirane slike.

Nato na izrezano sliko izpišite naključno generirano besedilo:

uporabite male in velike črke, števke ter presledke
naključno spreminjajte pisavo (lahko znotraj ene družine pisav), velikost in debelino
za teksturo besedila uporabite naključno izbrano fotografijo (ne enako kot za ozadje)
En izmed bolj enostavnih načinov prenosa teksture je ta, da najprej ustvarite masko, kje se besedilo nahaja. Za ustvarjanje maske najprej ustvarite črno sliko velikosti 256 x 256, nato pa na njo izpišete besedilo v beli barvi. Dodatno lahko obdržite samo prvi kanal in zavržete preostala dva. Nato uporabite masko, da iz naključno izbrane fotografije prenesete barvo pikslov na ozadje generirane slike. Na tak način ste prenesli samo tiste piksle, kjer se nahaja besedilo. Pravkar generirano masko pa uporabite na izhodu nevronske mreže.

# Maska velikost ozadja
target = np.zeros_like(image_bg)

# Izpis teksta na naključno pozicijo
target = cv.putText(target, text, position, font_face_rnd, font_scale, [255, 255, 255], thickness, cv.LINE_AA)

# Obdržimo samo 1 kanal
target = target[:, :, 0:1].astype(np.float32) / 255.
Tak način uporabe maske deluje v redu, vendar zavrže robne piksle besedila, ki so rezultat “anti aliasinga”. Robovi besedila so posledično zelo ostri in testiranje na realnih fotografijah, kjer robovi niso ostri, lahko deluje slabše. Delovanje na realnih fotografijah lahko izboljšate, če fotografijo, ki jo uporabite kot teksturo besedila, linearno interpolirate s fotografijo ozadja. Uporabite lahko naslednjo enačbo:

I = (1 - M) * B + M * F

Kjer je:
I - umetno generirana slika z besedilom
B - fotografija ozadja
F - fotografija teskture besedila
M - maska
# Aplikacija maske s teksturo teksta
image_bg = (1 - target) * image_bg + target * image_text
image_bg = image_bg.astype(np.float32) / 255.
Za še višjo raznolikost učnih podatkov uporabite tudi augmentacijo umetno generiranih slik. Primeri augmentacije so rotacija, skaliranje, vertikalna in horizontalna preslikava. Nekatere vrste augmentiranja vam lahko na sliko vnesejo črne ali interpolirane piksle na robovih. Pri rotaciji na primer, se deli slike, ki padejo izven rotirane slike obrežejo, novo prispeli piksli pa so lahko črni ali kako drugače interpolirani. Oboje je neželeno, saj zmanjša število uporabnih pikslov in posledično informacij za učenje. Ta pojav najlažje odpravite tako, da generirate slike velikosti 512 x 512 pikslov, ter jih po augmentaciji obrežete na 256 x 256 pikslov.

Vhod v nevronsko mrežo bo trikanalna barvna slika velikosti 256 x 256 z naključno izbrano fotografijo ozadja in naključnim besedilom v naključno izbrani fotografiji teksture. Izhod iz nevronske mreže bo enokanalna črno-bela slika velikosti 256 x 256, kjer bodo vrednosti 1 na mestih, kjer je na vhodni sliki besedilo in vrednosti 0 na mestih, kjer na vhodni sliki ni besedila.

Pred samim učenjem vrednosti vhodnih slik ustrezno skalirajte med 0 in 1, ter preuredite vrstni red dimenzij, da bo ustrezal tistemu, ki ga pričakuje ogrodje za učenje.

V poročilo zapišite naslednje lastnosti generiranih slik:

širina slik
višina slik
število učnih slik
število testnih slik
V poročilu prav tako izrišite nekaj umetno generiranih slik.

Gradnja nevronske mreže

Struktura nevronske mreže se zgleduje po arhitekturi U-Net. Rahlo je zmanjšano število slojev in filtrov, kar posledično pomeni hitrejše učenje:

Down blok (32, 3x3) → Down blok (64, 3x3) → Down blok (128, 3x3)
Conv2d (256, 3x3) → BatchNorm2d → ReLU
Conv2d (256, 3x3) → BatchNorm2d → ReLU
Up blok (128, 3x3) → Up blok (64, 3x3) → Up blok (32, 3x3)
Conv2d (1, 1x1)
Down blok ima naslednjo strukturo:

Conv2d → BatchNorm2d → ReLU
Conv2d → BatchNorm2d → ReLU
MaxPool2d (2x2, 2x2)
Up blok ima naslednjo strukturo:

ConvTranspose2d (2x2, 2x2)
Conv2d → BatchNorm2d → ReLU
Conv2d → BatchNorm2d → ReLU
Nevronska mreža ima poleg običajnih zaporednih povezav naprej tudi dodatne povezave, ki povežejo Down blok in Up blok z enakim številom konvolucijskih filtrov. V splošnem se Down blok z indeksom i poveže z Up blokom z indeksom i. Povezava je zgrajena tako, da vrednosti znotraj posameznega Down bloka pred uporabo sloja MaxPool2d (izhod drugega sloja ReLU), prenesemo v ustrezen Up blok, kjer jih konkateniramo z vrednostmi, ki jih vrne sloj ConvTranspose2d (vhod prvega sloja 2xConv).

Pri posamezni konvoluciji pazite, da bo širina in višina izhodnega tenzorja enaka širini in višini vhodnega tenzorja, saj želimo slike nespremenjenih velikosti. To najlažje dosežete z nastavitvijo parametra “padding” na ustrezno vrednost.

Vizualna predstavitev modificirane arhitekture nevronske mreže:



Povezava do izvorne arhitekture modela

 

Zadnji sloj nevronske mreže se lahko med implementacijami razlikuje:

V zadnjem sloju imamo konvolucijo z dvema filtroma velikosti 1x1, ki vrne sliko z dvema kanaloma, in softmax aktivacijsko funkcijo (torch.nn.functional.softmax ali tf.keras.activations.softmax). Kot metriko izgube uporabimo Cross-Entropy (torch.nn.CrossEntropyLoss ali tf.keras.losses.CategoricalCrossentropy). Ta način se običajno uporablja pri klasifikaciji več kot dveh razredov.
V zadnjem sloju imamo konvolucijo z enim filtrom velikosti 1x1, ki vrne sliko z enim kanalom, in sigmoidno aktivacijsko funkcijo (torch.nn.functional.sigmoid ali tf.keras.activations.sigmoid). Kot metriko izgube uporabimo Binary Cross-Entropy (torch.nn.BCELoss ali tf.keras.losses.BinaryCrossentropy). Ta način se običajno uporablja pri klasifikaciji dveh razredov.
V zadnjem sloju imamo konvolucijo z enim filtrom velikosti 1x1, ki vrne sliko z enim kanalom. Kot metriko izgube uporabimo Binary Cross-Entropy with Logits (torch.nn.BCEWithLogitsLoss ali tf.keras.losses.BinaryCrossentropy(from_logits=True)), ki interno združi sigmoidno aktivacijsko funkcijo in metriko izgube Binary Cross-Entropy. Ta način se običajno uporablja pri klasifikaciji dveh razredov in je bolj numerično stabilen kot 2. način.
V zgornji predstavitvi arhitekture nevronske mreže je predpostavljena uporaba 3. načina.

V poročilu prikažite strukturo zgrajene nevronske mreže. V pomoč vam je lahko na primer funkcija torchinfo.summary.

Učenje nevronske mreže

Za učenje nevronske mreže lahko uporabite naslednje parametre ali pa njihove vrednosti postavite na vam smiselne:

Optimizacijski algoritem: Adam
Hitrost učenja: 0.0001
Metrika izgube: Binary Cross-Entropy with Logits
Število epoh: min. 1000
Velikost batch: 32
Širina slik: 256
Višina slik: 256
Število slik na epoho: min. 1024
Število validacijskih slik na epoho: min. 256
Dobre rezultate bi morali doseči že pri nekje 500 epohah (pri 1024 slikah na epoho), vendar boste z višjim številom epoh dosegli boljše rezultate.

V poročilo zapišite vse uporabljene parametre učenja.

Izrišite graf napake učenja (loss) in graf validacijske napake (val loss).

Testiranje naučene nevronske mreže

Naučeno nevronsko mrežo testirajte na 1000 umetno generiranih slikah.

Delovanje mreže ovrednotite s pomočjo metrik IoU (Intersection over Union) in Dice koeficienta.

Metriki IoU (Intersection over Union) pravimo tudi Jaccard-ov indeks in se izračuna z naslednjo enačbo:

IOU = |A ∩ B| ÷ |A ∪ B| = TP ÷ (TP + FP + FN)

Dice koeficientu pravimo tudi Sørensen–Dice koeficient ali ocena F1 in se izračuna z naslednjo enačbo:

Dice koeficient = 2 × |A ∩ B| ÷ (|A| + |B|) = 2 × TP ÷ (2 × TP + FP + FN)

Vrednosti posamezne metrike izračunamo za vsak razred posebej in jih nato povprečimo s številom razredov. Vrednosti posamezne metrike so med 0 in 1.

Poročajte povrečno vrednost in standardni odklon za posamezno metriko.

Delovanje metode prikažite tudi na treh izbranih fotografijah, ki vsebujejo poljubno besedilo.

Oddaja naloge

Poročilo in program naj bo izdelan v Jupyther Notebook (.ipynb), zaledne funkcije lahko implementirate v ločenih .py datotekah. Na sistem oddajte datoteko naloga.zip, ki naj vsebuje porocilo.ipynb datoteko in njeno .html različico (izvoz). Prav tako oddajte vse lastne datoteke, ki se nanašajo na delovanje programa (.py datoteke). Ostalih datotek ne oddajajte! Poročilo naj v celicah vsebuje morebitne odgovore na zastavljena vprašanja, vse potrebne grafe ali slike ter ustrezno komentirane poglavitne dele kode programa. V primeru nespoštovanja predpisane oblike poročanja bo naloga zavrnjena in ocenjena z 0 točkami.