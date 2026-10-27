Électronique · AOP en régime non linéaire

# Les astables

Un astable n'a aucun état stable : sa sortie bascule toute seule, sans entrée, entre Vsat+ et Vsat−. Ce cours reprend le montage à AOP du cours manuscrit. Chaque formule est démontrée ligne par ligne, puis vérifiée sur des courbes.

Plan

1. 0[Rappels](#rappels)
2. 1[Le montage](#montage)
3. 2[Les deux seuils du trigger](#seuils)
4. 3[Le condensateur, phase par phase](#condo)
5. 4[Durées et période](#duree)
6. 5[Cas symétrique](#symetrique)
7. 6[Régler le rapport cyclique : diodes](#diodes)
8. 7[Solution finale : potentiomètre](#pot)
9. 8[Application : vitesse d'un moteur](#moteur)
10. 9[Récapitulatif](#recap)

## 0Rappels

Quatre résultats à avoir en tête avant de commencer.

### Rappel 1Un AOP idéal

Les courants d'entrée sont nuls. La grandeur qui commande la sortie est la différence ε = e+ − e−.

### Rappel 2Saturation

Sans réaction négative, ou avec une réaction positive, l'AOP est saturé : la sortie ne prend que deux valeurs.

ε > 0 ⇒ Vs = Vsat+ et ε \< 0 ⇒ Vs = Vsat−

Le basculement a lieu au moment où ε = 0, c'est-à-dire quand e+ = e−. Toute la suite revient à repérer cet instant.

### Rappel 3Bistable, monostable, astable

| Circuit | États stables | Comportement |
| --- | --- | --- |
| Bistable | 2 | Reste dans un état jusqu'à une impulsion qui le fait changer. Il mémorise la dernière impulsion. |
| Monostable | 1 | Une impulsion le fait passer dans un état quasi-stable. Il revient seul à l'état stable après une durée fixée par un réseau RC. |
| Astable | 0 | Aucun état ne dure. Il oscille seul entre les deux saturations, sans signal d'entrée. |

### Rappel 4Charge d'un condensateur à travers une résistance

Si une tension continue charge C à travers R, la tension Vc va de sa valeur initiale vers sa valeur finale selon :

Vc(t) = Vci + (Vcf − Vci) · (1 − e−(t−t0)/RC)

La constante de temps est τ = RC. Après une durée τ, le condensateur a parcouru 63 % de l'écart entre sa valeur initiale et sa valeur finale. Cette courbe est exponentielle : rapide au début, de plus en plus lente ensuite.

## 1Le montage

**Figure 1** · Montage astable. Le cadre pointillé est le trigger inverseur. R et C ferment une boucle de retard sur l'entrée (−).

### Les piècesCe que fait chaque élément

| Élément | Rôle |
| --- | --- |
| AOP | Fonctionne en comparateur. Sa sortie vaut Vsat+ ou Vsat−, jamais une valeur intermédiaire. |
| R2 | Ramène la sortie sur l'entrée (+). C'est une **réaction positive** : elle force la saturation et fabrique l'hystérésis. |
| R1 et Vref | R1 relie l'entrée (+) à la source Vref. Avec R2, elle forme un pont diviseur qui fixe les deux seuils de basculement. |
| R | Relie la sortie à l'entrée (−). Elle règle la vitesse de charge et de décharge du condensateur. |
| C | Garde en mémoire l'histoire du circuit. Sa tension Vc = e− est celle qu'on compare au seuil. |

Le cadre rouge en pointillé du cours s'appelle un **trigger inverseur** : un comparateur à hystérésis dont le signal à comparer entre sur l'entrée (−). Ici ce signal n'est pas extérieur, c'est la tension du condensateur.

### IdéePourquoi ça oscille

1. La sortie est à Vsat+. Le condensateur se charge à travers R vers Vsat+, donc e− monte.
2. Quand e− atteint le seuil e+, la sortie bascule à Vsat−. Au même moment, le seuil e+ change de valeur.
3. Le condensateur se décharge à travers R vers Vsat−, donc e− descend.
4. Quand e− atteint le nouveau seuil, la sortie rebascule à Vsat+. Retour à l'étape 1.

Aucun état ne peut durer : la tension que le condensateur cherche à atteindre (la sortie) est toujours située au-delà du seuil. Le condensateur finit donc toujours par le franchir. C'est exactement la définition de l'astable.

## 2Étape 1 : les deux seuils du trigger

### 2.1Tension sur l'entrée (+)

Le courant d'entrée de l'AOP est nul. R2 et R1 sont donc en série, entre la sortie Vs et la source Vref, et traversées par le même courant :

i = Vs − VrefR1 + R2

La tension en (+) est celle de la source plus la chute dans R1 :

e+ = Vref + R1 · Vs − VrefR1 + R2 = R1 Vs + R2 VrefR1 + R2

Le théorème de superposition donne le même résultat : e+ est la somme de la contribution de Vs (pont diviseur, facteur R1/(R1+R2)) et de celle de Vref (facteur R2/(R1+R2)).

### 2.2Deux seuils, pas un seul

Le seuil e+ dépend de Vs, qui ne prend que deux valeurs. Il y a donc deux seuils, notés comme dans le cours V1 et V2 :

V1 = R1 Vsat+ + R2 VrefR1 + R2 seuil quand Vs = Vsat+

V2 = R1 Vsat− + R2 VrefR1 + R2 seuil quand Vs = Vsat−

Comme Vsat+ > Vsat−, on a toujours V1 > V2. L'écart est la **largeur d'hystérésis** :

V1 − V2 = R1 (Vsat+ − Vsat−)R1 + R2

### 2.3Quand a lieu chaque basculement

Le cours le formule ainsi. V1 est le moment du basculement de Vsat+ vers Vsat− : à cet instant e− = e+ alors que Vs = Vsat+, ce qui donne bien l'expression de V1 ci-dessus. De même, V2 est le moment du basculement de Vsat− vers Vsat+, avec Vs = Vsat−.

| Sortie avant | Seuil actif | Basculement quand | Sortie après |
| --- | --- | --- | --- |
| Vsat+ | e+ = V1 | e− monte et atteint V1 | Vsat− |
| Vsat− | e+ = V2 | e− descend et atteint V2 | Vsat+ |

**Figure 2** · Caractéristique Vs = f(e−) du trigger inverseur (ici R1 = R2, Vref = 3 V, Vsat = 12 V, soit V1 = 7,5 V et V2 = −4,5 V). On suit le cycle dans le sens des flèches.

Lecture de la figure : pour e− croissant, la sortie reste à Vsat+ jusqu'à V1, puis tombe. Pour e− décroissant, elle reste à Vsat− jusqu'à V2, puis remonte. Entre V2 et V1, la valeur de la sortie dépend du chemin parcouru : c'est l'hystérésis.

**Remarque du prof.**L'hystérésis est ce qui rend l'oscillation possible. Avec un seul seuil, la sortie basculerait, ferait varier e− en sens inverse et rebasculerait aussitôt. Avec deux seuils, il faut une vraie course de Vc entre V2 et V1 avant le basculement suivant.

## 3Étape 2 : le condensateur, phase par phase

### 3.1L'équation de la charge

Tant que la sortie ne bascule pas, Vs est une constante. La loi des mailles sur R et C s'écrit, avec iC = C dVc/dt :

Vs = R · C dVcdt + Vc

C'est une équation du premier ordre à second membre constant. Sa solution, avec Vc(t0) comme condition initiale, est :

Vc(t) = Vs + (Vc(t0) − Vs) · e−(t−t0)/RC

Vérification : à t = t0 on retrouve Vc(t0), et pour t → ∞ on tend vers Vs. C'est le Rappel 4 réécrit avec Vci = Vc(t0) et Vcf = Vs.

### 3.2La version du cours, avec le courant

Le cours oriente le courant i du condensateur vers la sortie et écrit :

e−(t) = Vsat+ + R · i(t) avec i(t) = A · e−(t−t0)/RC

La constante A se trouve avec la condition initiale. À t0, e−(t0) = Vc(t0) = Vsat+ + R · i(t0), donc :

A = i(t0) = Vc(t0) − Vsat+R ⇒ e−(t) = Vsat+ + (Vc(t0) − Vsat+) · e−(t−t0)/RC

C'est exactement la formule du 3.1 avec Vs = Vsat+. Le signe de A est négatif pendant la charge, parce que le condensateur est en dessous de Vsat+ : le courant réel circule dans le sens opposé à celui choisi.

### 3.3Les trois phases

| Phase | Intervalle | Vs | e+ | Vc évolue | Fin de phase |
| --- | --- | --- | --- | --- | --- |
| 1 | t0 \< t \< t1 | Vsat+ | V1 | croît, de Vc(t0) vers Vsat+ | e− > e+ |
| 2 | t1 \< t \< t2 | Vsat− | V2 | décroît, de V1 vers Vsat− | e− \< e+ |
| 3 | t2 \< t \< t3 | Vsat+ | V1 | croît, de V2 vers Vsat+ | e− > e+ |

### Phase 1t0 \< t \< t1 : charge vers Vsat+

On suppose Vs = Vsat+ à t0. Le seuil est donc e+ = V1.

Vc(t) = e−(t) = Vc(t0) + (Vsat+ − Vc(t0)) · (1 − e−(t−t0)/RC)

C'est une fonction croissante, puisque Vsat+ > Vc. Elle monte jusqu'à ce que e− > e+ = V1. À cet instant, **basculement de Vsat+ vers Vsat−**.

### Phase 2t1 \< t \< t2 : décharge vers Vsat−

La sortie vaut Vsat−, le seuil est e+ = V2. À t1, le condensateur vaut Vc(t1) = V1, puisque c'est lui qui a déclenché le basculement. On reporte dans la formule générale :

Vc(t) = e−(t) = V1 + (Vsat− − V1) · (1 − e−(t−t1)/RC)

Ici Vsat− − V1 est négatif : la fonction est décroissante. Elle descend jusqu'à e− \< e+ = V2. À cet instant, **basculement de Vsat− vers Vsat+**.

### Phase 3t2 \< t \< t3 : recharge vers Vsat+

La sortie est revenue à Vsat+, le seuil est e+ = V1. À t2, Vc(t2) = V2 :

Vc(t) = V2 + (Vsat+ − V2) · (1 − e−(t−t2)/RC)

La fonction croît de V2 jusqu'à V1 à l'instant t3, puis tout recommence : la phase 3 est identique à la phase 1, avec un point de départ différent seulement à la mise sous tension.

**Figure 3** · Tension Vc = e− (bleu) et sortie Vs (rouge), avec R1 = R2, Vref = 0, Vsat = 12 V, temps en unités de RC. À la mise sous tension le condensateur est déchargé : Vc(t0) = 0, donc la phase 1 est plus courte que les suivantes. Le régime périodique commence en t1.

Deux repères pour lire la courbe. Les droites pointillées horizontales sont les seuils V1 et V2 : Vc oscille entre elles et ne va jamais jusqu'à ±Vsat. Chaque morceau de la courbe est un arc d'exponentielle, qui vise Vsat+ ou Vsat− mais est interrompu en route.

## 4Étape 3 : durées et période

On cherche le temps que met le condensateur pour aller d'un seuil à l'autre. Il suffit d'écrire la condition de fin de phase dans la formule de la phase.

### 4.1Durée de la phase 3, sortie à Vsat+

À t3, Vc(t3) = V1. Avec la forme Vs + (Vc(t0) − Vs)e… :

V1 = Vsat+ + (V2 − Vsat+) · e−(t3−t2)/RC

On isole l'exponentielle :

e−(t3−t2)/RC = Vsat+ − V1Vsat+ − V2

On passe au logarithme. Comme ln(1/x) = −ln x, le signe moins disparaît en inversant la fraction :

t3 − t2 = RC · ln Vsat+ − V2Vsat+ − V1 durée à Vsat+

### 4.2Durée de la phase 2, sortie à Vsat−

À t2, Vc(t2) = V2, avec Vs = Vsat− et un départ à V1 :

V2 = Vsat− + (V1 − Vsat−) · e−(t2−t1)/RC ⇒ e−(t2−t1)/RC = Vsat− − V2Vsat− − V1

t2 − t1 = RC · ln Vsat− − V1Vsat− − V2 durée à Vsat−

Contrôle de signe : V1 > V2, donc chaque fraction est supérieure à 1 et chaque logarithme est positif. Une durée négative signalerait une erreur de calcul.

### 4.3Période

La période se mesure entre deux basculements de même sens, par exemple t1 et t3 :

T = t3 − t1 = (t3 − t2) + (t2 − t1) = RC · ln (Vsat+ − V2)(Vsat− − V1)(Vsat+ − V1)(Vsat− − V2)

La somme de deux logarithmes est le logarithme du produit, d'où cette écriture.

### 4.4Rapport cyclique

Le rapport cyclique est la fraction de la période passée à l'état haut :

δ = TonT = t3 − t2T = ln \[(Vsat+ − V2)/(Vsat+ − V1)\]ln \[(Vsat+ − V2)(Vsat− − V1) / ((Vsat+ − V1)(Vsat− − V2))\]

## 5Cas symétrique : Vsat+ = −Vsat− = Vsat

En pratique les deux saturations sont opposées. Le calcul se simplifie beaucoup avec trois notations :

a = R1R1 + R2 , k = 1 + a1 − a = 1 + 2R1R2 , ρ = VrefVsat

Le passage de (1+a)/(1−a) à 1 + 2R1/R2 vient de 1 + a = (2R1 + R2)/(R1 + R2) et 1 − a = R2/(R1 + R2).

### 5.1Les seuils

Comme R2/(R1+R2) = 1 − a, les seuils s'écrivent :

V1 = a Vsat + (1 − a) Vref et V2 = −a Vsat + (1 − a) Vref

### 5.2Les quatre différences du log

Avec Vsat− = −Vsat, la période du 4.3 devient :

T = RC · ln (Vsat − V2)(Vsat + V1)(Vsat − V1)(Vsat + V2)

On calcule chaque facteur en mettant Vsat(1 − a) en évidence :

Vsat − V2 = Vsat(1−a)(k − ρ) Vsat + V1 = Vsat(1−a)(k + ρ)

Vsat − V1 = Vsat(1−a)(1 − ρ) Vsat + V2 = Vsat(1−a)(1 + ρ)

Exemple de vérification : Vsat + V1 = Vsat(1 + a) + (1−a)ρVsat = Vsat(1−a)(k + ρ), car 1 + a = (1−a)k. Le facteur Vsat(1−a) se simplifie entre le haut et le bas de la fraction.

### 5.3Résultat général

Le haut contient (k − ρ)(k + ρ), le bas (1 − ρ)(1 + ρ). On applique (x−y)(x+y) = x² − y², comme dans le cours :

T = RC · ln (1 + 2R1/R2)² − (Vref/Vsat)²1 − (Vref/Vsat)²

Les deux durées partielles se lisent aussi directement :

t3 − t2 = RC · ln k − ρ1 − ρ et t2 − t1 = RC · ln k + ρ1 + ρ

**Condition d'oscillation.** Les fractions du log doivent être positives, ce qui impose |Vref| \< Vsat. Physiquement, il faut que les seuils restent dans la plage de saturation : V1 \< Vsat+ et V2 > Vsat−. Sinon le condensateur n'atteindrait jamais le seuil et la sortie resterait bloquée.

### 5.4Le cas du cours : Vref = 0

Alors ρ = 0, et V1 = −V2 = a Vsat : les deux seuils sont symétriques.

T = 2 · RC · ln(1 + 2R1/R2) t3−t2 = t2−t1 = RC · ln(1 + 2R1/R2)

δ = RC · ln(1 + 2R1/R2)2 · RC · ln(1 + 2R1/R2) = 0,5

Le signal est un **carré parfaitement symétrique**, et δ = 0,5 quelles que soient les valeurs de R, C, R1, R2. La fréquence se règle, le rapport cyclique non. C'est le point de départ de la suite du cours.

**Exemple chiffré.** R = 100 kΩ, C = 1 µF, R1 = R2 : RC = 0,1 s et 1 + 2R1/R2 = 3. Alors T = 2 × 0,1 × ln 3 ≈ 0,22 s, soit une fréquence d'environ 4,55 Hz. Avec Vsat = 12 V, les seuils sont V1 = +6 V et V2 = −6 V.

### EssayerManipuler les paramètres

La courbe ci-dessous est recalculée à partir des équations de la phase, et la période est mesurée sur la courbe elle-même. Elle doit coïncider avec la formule du 5.3 pour toute valeur admissible.

R1 10 kΩ R2 10 kΩ Vref 0 V

V1 =

V2 =

T formule =

T sur la courbe =

δ =

**Figure 4** · Exploration de l'astable. Vsat = 12 V est fixe, le temps est en unités de RC. Avec Vref ≠ 0, le rapport cyclique s'écarte de 0,5 mais la période change aussi : ce n'est donc pas un moyen de régler δ seul.

## 6Régler le rapport cyclique : première modification avec des diodes

Avec Vref = 0, on a δ = 0,5 constant. Pour obtenir un autre rapport cyclique, il faut que la charge et la décharge du condensateur n'aient pas la même constante de temps. On le fait en leur donnant deux chemins différents, séparés par deux diodes montées en sens opposés : D avec la résistance R, et D' avec la résistance R', toutes deux en parallèle à la place de l'ancienne résistance R.

| Phase | Sortie | Condensateur | Diode passante | Résistance |
| --- | --- | --- | --- | --- |
| t2 \< t \< t3 | Vsat+ | se charge | D passante, D' bloquée | R |
| t1 \< t \< t2 | Vsat− | se décharge | D' passante, D bloquée | R' |

Le courant ne peut traverser qu'une diode à la fois, dans le sens qui lui est permis. Selon que la sortie est haute ou basse, il passe donc par l'une ou par l'autre.

### 6.1Nouvelles durées (Vref = 0)

La formule du 4.2 reste vraie en remplaçant R par R'. On simplifie le log avec Vsat− = −Vsat :

t2 − t1 = R'C · ln −Vsat − (R1Vsat/(R1+R2))−Vsat + R1Vsat/(R1+R2) = R'C · ln 2R1 + R2R2

t2 − t1 = R'C · ln(1 + 2R1/R2) t3 − t2 = RC · ln(1 + 2R1/R2)

T = (R + R')C · ln(1 + 2R1/R2) δ = RR + R'

Ces calculs supposent des diodes idéales. Avec des diodes réelles, la chute d'environ 0,6 V décale légèrement les durées.

### 6.2Pourquoi ce n'est pas encore satisfaisant

| R' | δ | T |
| --- | --- | --- |
| R/3 | 0,75 | (4/3) · RC · ln(1 + 2R1/R2) |
| R | 0,50 | 2 · RC · ln(1 + 2R1/R2) |
| 3R | 0,25 | 4 · RC · ln(1 + 2R1/R2) |

Chaque fois qu'on règle δ en changeant R', la période T change aussi, car T contient R + R'. Pour un moteur, c'est gênant : on voudrait faire varier la vitesse sans toucher à la fréquence. Il faut une autre solution.

## 7Solution finale : un potentiomètre entre les deux diodes

On remplace R et R' par un potentiomètre de résistance totale R. Le curseur, repéré par la fraction K avec 0 \< K \< 1, partage cette résistance en deux parties dont la somme est constante :

- le chemin de charge, par D, traverse la partie K · R ;
- le chemin de décharge, par D', traverse la partie (1 − K) · R.

On applique les résultats du 6.1 avec R → K R et R' → (1 − K) R :

T = (K R + (1 − K) R) · C · ln(1 + 2R1/R2) = R C · ln(1 + 2R1/R2)

δ = K RK R + (1 − K) R = K

La somme K R + (1−K) R vaut toujours R : la période ne dépend plus de la position du curseur. Le rapport cyclique, lui, vaut exactement K et ne dépend plus d'aucune valeur de résistance. On règle δ avec le curseur sans changer T.

**Figure 5** · Sortie Vs pour trois positions du curseur. La période est la même, seule la largeur de l'état haut change : Ton = K · T.

## 8Application : variation de vitesse d'un moteur

La sortie Vs de l'astable est un signal MLI (modulation de largeur d'impulsion, PWM en anglais). Elle commande la grille d'un transistor NMOS placé en série avec le moteur, alimenté sous +Vcc. Le NMOS laisse passer le courant quand la tension de grille est positive, c'est-à-dire pendant l'état haut Vs = Vsat+. Le transistor joue le rôle d'un interrupteur commandé :

0 \< t \< T0 : U(t) = Vcc moteur alimenté

T0 \< t \< T : U(t) = 0 V moteur coupé

Ici T0 est la durée de l'état haut, c'est-à-dire Ton = δ T = K T.

### 8.1Tension moyenne

Le moteur réagit à la valeur moyenne de la tension, rappelée par la définition de la moyenne sur une période :

Vmoy = 1T ∫0T U(t) dt = 1T ∫0T0 Vcc dt = T0T Vcc

La deuxième intégrale s'arrête à T0 parce que U = 0 ensuite. On obtient :

Vmoy = δ · Vcc = K · Vcc

La tension moyenne est proportionnelle à la position K du curseur. Tourner le potentiomètre fait varier la vitesse du moteur de façon linéaire, entre 0 et la vitesse maximale à K = 1.

**Figure 6** · Gauche : tension U(t) aux bornes du moteur pour K = 0,25 et K = 0,75, avec la valeur moyenne en pointillé. Droite : Vmoy = K · Vcc.

**Remarque du prof.**Pour que le moteur « voie » la moyenne, la période T doit être petite devant la constante de temps mécanique du moteur. L'exemple chiffré du 5.4, à environ 4,55 Hz, donnerait des à-coups. En pratique on choisit R et C pour viser quelques centaines de hertz : avec R = 100 kΩ, C = 10 nF et R1 = R2, le montage du 7 donne T = RC ln 3 ≈ 1,1 ms, soit environ 910 Hz. Le moteur étant inductif, on ajoute en général une diode de roue libre en parallèle pour protéger le transistor à la coupure.

## 9Récapitulatif

| Grandeur | Expression |
| --- | --- |
| Tension en (+) | e+ = (R1Vs + R2Vref) / (R1 + R2) |
| Seuils | V1 = (R1Vsat+ + R2Vref) / (R1+R2), V2 = (R1Vsat− + R2Vref) / (R1+R2) |
| Condensateur | Vc(t) = Vs + (Vc(t0) − Vs) e−(t−t0)/RC |
| Durée à Vsat+ | t3 − t2 = RC ln\[(Vsat+ − V2) / (Vsat+ − V1)\] |
| Durée à Vsat− | t2 − t1 = RC ln\[(Vsat− − V1) / (Vsat− − V2)\] |
| Période, cas symétrique | T = RC ln\[((1 + 2R1/R2)² − (Vref/Vsat)²) / (1 − (Vref/Vsat)²)\] |
| Vref = 0 | T = 2RC ln(1 + 2R1/R2), δ = 0,5 |
| Deux diodes R, R' | T = (R + R')C ln(1 + 2R1/R2), δ = R / (R + R') |
| Potentiomètre | T = RC ln(1 + 2R1/R2), δ = K |
| Moteur | Vmoy = K · Vcc |

### MéthodeÀ savoir refaire seul, dans cet ordre

1. Écrire e+ par le pont diviseur, puis les deux seuils V1 et V2.
2. Pour chaque phase, noter Vs, le seuil actif, la valeur de départ de Vc et la valeur visée.
3. Écrire Vc(t) avec la solution de l'équation du premier ordre.
4. Écrire la condition de fin de phase (Vc = V1 ou Vc = V2) et résoudre en t avec un logarithme.
5. Additionner les deux durées pour obtenir T, et diviser la durée à l'état haut par T pour δ.
6. Contrôler les signes : chaque durée doit être positive et |Vref| \< Vsat.

Les formules de ce cours ont été vérifiées par simulation numérique de l'équation du condensateur avec les deux seuils, pour plusieurs jeux de paramètres dont des cas avec Vref ≠ 0 : la période et le rapport cyclique concordent avec les expressions établies.