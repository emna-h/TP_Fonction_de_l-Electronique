# Électronique : bistables, monostables, astables et exercices corrigés

Fiche de révision regroupant tout ce qui a été expliqué : application « barrière de parking », bistable, astables (cours complet démontré), calculs de e⁺, exercices 2, 3 et exercice du bistable.

Les figures sont dans le même dossier que ce fichier : `triggers.png`, `ex2.png`, `wave.png`, `motor.png`.

## Sommaire

1. Application : commande de levée d'une barrière de parking
2. Le bistable (cours)
3. Les astables (cours complet)
4. Calcul de e⁺ pour un nœud à trois résistances
5. La 2ᵉ partie du circuit : V_D = (2/3)(V_c + E)
6. Pourquoi ce n'est pas un trigger non inverseur
7. Exercice 2 : monostable déclenché par un front
8. Exercice 3 : modulateur de largeur d'impulsion
9. Exercice du bistable : correction détaillée
10. Formulaire
11. Points à confirmer

**Conventions.** Les AOP sont idéaux : courants d'entrée nuls, et ε = e⁺ − e⁻ commande la sortie. En saturation, ε > 0 donne Vs = Vsat⁺ et ε < 0 donne Vs = Vsat⁻. Le basculement a lieu quand e⁺ = e⁻.

---

## 1. Application : commande de levée d'une barrière de parking

### 1.1 Cahier des charges

| Paramètre | Valeur | Signification |
|---|---|---|
| T↑ | 10 s | durée de montée (levée) |
| T_H | 30 s | durée de maintien en position haute |
| T_f | 10 s | durée de fermeture (descente) |

Cycle : lever (10 s), rester ouverte (30 s), descendre (10 s).

### 1.2 Architecture

```
Carte magnétique → [BLOC 1 : comparateur] → [BLOC 2 : dérivateur] → { MONO 1 → NMOS → relais 1 → moteur (lever)
 (sinusoïde)         sinus → créneau         créneau → impulsions   { MONO 2 → PMOS → relais 2 → moteur (descendre)
```

Un seul passage de carte produit deux impulsions séparées par T_H. La première commande la montée, la seconde la descente.

L'écriture manuscrite est parfois ambiguë (brochage des AOP, sens des diodes). Les points d'interprétation sont signalés.

### 1.3 Bloc 1 : sinusoïde → créneaux (comparateur simple inverseur)

- La carte magnétique génère un signal sinusoïdal. L'interrupteur DI se ferme à l'insertion de la carte.
- La pile Ve est la tension de référence (seuil) du comparateur.
- L'AOP est en boucle ouverte : sa sortie vaut +V_sat ou −V_sat.

Fonctionnement, le signal étant sur l'entrée (−) et Ve sur (+) :

```
e(t) < Ve  →  ε > 0  →  Vs = +Vsat
e(t) > Ve  →  ε < 0  →  Vs = −Vsat
```

La sortie bascule à chaque fois que la sinusoïde croise Ve. Ve fixe la largeur du créneau haut, donc l'intervalle T_H entre les deux impulsions finales :

- Ve ↑ ⇒ T_H ↑
- Ve ↓ ⇒ T_H ↓

### 1.4 Bloc 2 : créneaux → impulsions (dérivateur)

Un condensateur C en série à l'entrée et une résistance R en contre-réaction forment un dérivateur inverseur :

```
Vs = −R·C·dVe/dt
```

- Sur un niveau constant, la dérivée est nulle : Vs ≈ 0.
- Sur un front, la dérivée est très grande : impulsion (Dirac).
- Montage inverseur : front montant → impulsion négative ; front descendant → impulsion positive.

Résultat : une impulsion négative, puis, T_H plus tard, une impulsion positive.

### 1.5 Monostables MONO 1 et MONO 2

Ce sont des comparateurs à réaction positive avec un réseau R // C et une diode reliée à V_ref. Un monostable a un état stable et un état quasi-stable dont la durée dépend de R et C : T ≈ R·C·ln(…).

Principe : une impulsion de déclenchement fait basculer la sortie, le condensateur se charge à travers R, et quand sa tension atteint le seuil la sortie rebascule seule.

| | MONO 1 | MONO 2 |
|---|---|---|
| V_ref | < 0 | > 0 |
| Déclenché par | impulsion négative (la première) | impulsion positive (la seconde) |
| Condition | impulsion ≤ V_ref | impulsion ≥ V_ref |
| État stable | V_sat⁻ | V_sat⁺ |
| État quasi-stable | V_sat⁺ pendant T↑ = 10 s | V_sat⁻ pendant T_f = 10 s |

Chaque monostable ignore l'impulsion de signe opposé grâce à sa diode et au signe de V_ref.

### 1.6 Étage de puissance et moteur

Chaque MOS est un interrupteur commandé en série avec la bobine d'un relais alimentée en 5 V.

| | NMOS (voie MONO 1) | PMOS (voie MONO 2) |
|---|---|---|
| Conduit si | V_GS > V_th (grille à +V_sat) | V_GS < V_th négatif (grille à −V_sat) |
| Repos | grille à −V_sat, bloqué | grille à +V_sat, bloqué |
| Pendant l'impulsion | passant, bobine alimentée | passant, bobine alimentée |

Le moteur est alimenté par deux sources de polarités opposées, chacune connectée par un contact de relais différent. Le relais 1 le fait tourner dans le sens « lever », le relais 2 dans le sens « descendre ».

### 1.7 Déroulement dans le temps

| Étape | Événement | Effet |
|---|---|---|
| 1 | insertion de la carte, DI se ferme | la sinusoïde arrive au Bloc 1 |
| 2 | Bloc 1 compare à Ve | créneau de largeur T_H |
| 3 | Bloc 2 dérive | impulsion négative, puis positive T_H plus tard |
| 4 | impulsion négative ≤ V_ref (< 0) | MONO 1 bascule, NMOS passant, relais 1, la barrière monte |
| 5 | après T↑ = 10 s | MONO 1 revient au repos, barrière ouverte |
| 6 | attente T_H = 30 s | le véhicule passe |
| 7 | impulsion positive ≥ V_ref (> 0) | MONO 2 bascule, PMOS passant, relais 2, moteur en sens inverse |
| 8 | après T_f = 10 s | barrière fermée, fin du cycle |

À retenir : Ve règle T_H, les R et C des monostables règlent T↑ et T_f, les signes opposés de V_ref séparent les deux impulsions.

---

## 2. Le bistable (cours)

### 2.1 Montage

- AOP en boucle ouverte avec **réaction positive** : la sortie Vs est ramenée sur l'entrée (+) par le pont R₂ (série) / R₁ (vers la masse).
- Le signal (suite d'impulsions : une positive, puis une négative) est appliqué sur l'entrée (−).
- L'AOP est toujours saturé : Vs = +Vsat ou −Vsat.
- Deux états stables : le bistable mémorise la dernière impulsion reçue.

### 2.2 Seuils

```
e⁺ = R₁/(R₁+R₂) · Vs
```

| Sortie | Seuil e⁺ |
|---|---|
| Vs = +Vsat | + R₁/(R₁+R₂) · Vsat⁺ (seuil haut) |
| Vs = −Vsat | R₁/(R₁+R₂) · Vsat⁻ (seuil bas, négatif) |

La dépendance du seuil à la sortie crée l'hystérésis.

### 2.3 Déroulement

- **t = 0** (supposition) : Vs = Vsat⁺, e⁺ = R₁/(R₁+R₂)·Vsat⁺, e⁻ = 0 < e⁺ : état stable.
- **t = t₁**, impulsion positive : e⁻ > R₁/(R₁+R₂)·Vsat⁺ donne un basculement, Vs = Vsat⁻, et le seuil devient R₁/(R₁+R₂)·Vsat⁻. Quand l'impulsion disparaît, e⁻ = 0 > e⁺ : la sortie reste à Vsat⁻ (effet mémoire).
- **t = t₂**, impulsion négative : e⁻ < R₁/(R₁+R₂)·Vsat⁻ donne un basculement, Vs = Vsat⁺.

| Basculement | Condition | Sortie |
|---|---|---|
| t₁ | e⁻ > R₁/(R₁+R₂)·Vsat⁺ | +Vsat → −Vsat |
| t₂ | e⁻ < R₁/(R₁+R₂)·Vsat⁻ | −Vsat → +Vsat |

La largeur du créneau obtenu est t₂ − t₁ : l'écart entre les deux impulsions. Contrairement au monostable, il n'y a pas de temporisation interne.

---

## 3. Les astables (cours complet)

Un astable n'a aucun état stable : sa sortie bascule seule entre Vsat⁺ et Vsat⁻.

### 3.0 Rappels

| Circuit | États stables | Comportement |
|---|---|---|
| Bistable | 2 | reste dans un état jusqu'à une impulsion, mémorise |
| Monostable | 1 | une impulsion déclenche un état quasi-stable, retour seul |
| Astable | 0 | oscille seul sans signal d'entrée |

Charge d'un condensateur à travers R :

```
Vc(t) = Vci + (Vcf − Vci) · (1 − e^(−(t−t₀)/RC))        τ = R·C
```

### 3.1 Le montage

Éléments : R₂ ramène la sortie sur (+) (réaction positive, hystérésis) ; R₁ et V_ref fixent les seuils ; R relie la sortie à (−) ; C garde la mémoire, sa tension Vc = e⁻ est comparée au seuil. Le cadre rouge du cours est un **trigger inverseur**.

Pourquoi ça oscille :

1. Sortie à Vsat⁺, C se charge vers Vsat⁺, e⁻ monte.
2. e⁻ atteint le seuil e⁺ : bascule à Vsat⁻, le seuil change.
3. C se décharge vers Vsat⁻, e⁻ descend.
4. e⁻ atteint le nouveau seuil : rebascule à Vsat⁺.

La tension que C cherche à atteindre (la sortie) est toujours au-delà du seuil, donc il le franchit toujours.

### 3.2 Les deux seuils

Courant d'entrée nul : R₂ et R₁ sont en série entre Vs et V_ref.

```
i  = (Vs − Vref)/(R₁+R₂)
e⁺ = Vref + R₁·i = (R₁·Vs + R₂·Vref)/(R₁+R₂)
```

Deux seuils (notations du cours) :

```
V₁ = (R₁·Vsat⁺ + R₂·Vref)/(R₁+R₂)     quand Vs = Vsat⁺
V₂ = (R₁·Vsat⁻ + R₂·Vref)/(R₁+R₂)     quand Vs = Vsat⁻
V₁ − V₂ = R₁(Vsat⁺ − Vsat⁻)/(R₁+R₂)    (largeur d'hystérésis)
```

| Sortie avant | Seuil actif | Basculement quand | Sortie après |
|---|---|---|---|
| Vsat⁺ | V₁ | e⁻ monte et atteint V₁ | Vsat⁻ |
| Vsat⁻ | V₂ | e⁻ descend et atteint V₂ | Vsat⁺ |

Exemple (figure d'hystérésis du cours) : R₁ = R₂, V_ref = 3 V, Vsat = 12 V donne V₁ = 7,5 V et V₂ = −4,5 V. L'hystérésis est indispensable : avec un seul seuil, la sortie rebasculerait aussitôt.

### 3.3 Le condensateur, phase par phase

**Obtenir Vs = R·C·dVc/dt + Vc.** Loi des mailles entre la sortie et la masse :

```
Vs = v_R + Vc  avec  v_R = R·i  et  i = C·dVc/dt
⇒  Vs = R·C·dVc/dt + Vc
```

Le courant d'entrée de l'AOP étant nul, le même courant traverse R puis charge C. Avec τ = RC, l'équation τ·dVc/dt + Vc = Vs a pour solution (homogène + particulière) :

```
Vc(t) = Vs + (Vc(t₀) − Vs) · e^(−(t−t₀)/RC)
```

Vérification : en t₀ on retrouve Vc(t₀), pour t → ∞ on tend vers Vs.

**Forme du cours avec le courant :** e⁻(t) = Vsat⁺ + R·i(t), avec i(t) = A·e^(−(t−t₀)/RC) et A = i(t₀) = (Vc(t₀) − Vsat⁺)/R. On retrouve la même formule. A est négatif pendant la charge car le courant réel est opposé au sens choisi.

| Phase | Intervalle | Vs | e⁺ | Vc évolue | Fin de phase |
|---|---|---|---|---|---|
| 1 | t₀ < t < t₁ | Vsat⁺ | V₁ | croît vers Vsat⁺ | e⁻ > e⁺ |
| 2 | t₁ < t < t₂ | Vsat⁻ | V₂ | décroît de V₁ vers Vsat⁻ | e⁻ < e⁺ |
| 3 | t₂ < t < t₃ | Vsat⁺ | V₁ | croît de V₂ vers Vsat⁺ | e⁻ > e⁺ |

```
Phase 1 : Vc(t) = Vc(t₀) + (Vsat⁺ − Vc(t₀))·(1 − e^(−(t−t₀)/RC))
Phase 2 : Vc(t) = V₁ + (Vsat⁻ − V₁)·(1 − e^(−(t−t₁)/RC))        (Vc(t₁) = V₁)
Phase 3 : Vc(t) = V₂ + (Vsat⁺ − V₂)·(1 − e^(−(t−t₂)/RC))        (Vc(t₂) = V₂)
```

Dans le cas de la figure (R₁ = R₂, V_ref = 0), la phase 1 est plus courte, car C part de 0 et non de V₂. Le régime périodique commence en t₁. Vc oscille entre V₂ et V₁ sans atteindre ±Vsat.

### 3.4 Durées, période, rapport cyclique

Phase 3 : Vc(t₃) = V₁.

```
V₁ = Vsat⁺ + (V₂ − Vsat⁺)·e^(−(t₃−t₂)/RC)
e^(−(t₃−t₂)/RC) = (Vsat⁺ − V₁)/(Vsat⁺ − V₂)
t₃ − t₂ = RC · ln[(Vsat⁺ − V₂)/(Vsat⁺ − V₁)]
```

Phase 2 : Vc(t₂) = V₂.

```
V₂ = Vsat⁻ + (V₁ − Vsat⁻)·e^(−(t₂−t₁)/RC)
t₂ − t₁ = RC · ln[(Vsat⁻ − V₁)/(Vsat⁻ − V₂)]
```

Chaque fraction est supérieure à 1 (car V₁ > V₂) : un log négatif signalerait une erreur.

```
T = (t₃−t₂) + (t₂−t₁) = RC · ln[ (Vsat⁺−V₂)(Vsat⁻−V₁) / ((Vsat⁺−V₁)(Vsat⁻−V₂)) ]
δ = T_on/T = (t₃−t₂)/T
```

### 3.5 Cas symétrique Vsat⁺ = −Vsat⁻ = Vsat

Trois notations :

```
a = R₁/(R₁+R₂)        1 − a = R₂/(R₁+R₂)  (car les deux fractions ont pour somme 1)
k = (1+a)/(1−a) = 1 + 2R₁/R₂
ρ = Vref/Vsat
```

Passage de k à 1 + 2R₁/R₂ : 1 + a = (2R₁+R₂)/(R₁+R₂) et 1 − a = R₂/(R₁+R₂), les (R₁+R₂) se simplifient.

**Seuils :**

```
V₁ = a·Vsat + (1−a)·Vref  = Vsat·[ a + (1−a)ρ ]
V₂ = −a·Vsat + (1−a)·Vref = Vsat·[ −a + (1−a)ρ ]
```

**Période.** Avec Vsat⁻ = −Vsat, (−Vsat−V₁)/(−Vsat−V₂) = (Vsat+V₁)/(Vsat+V₂) (les deux signes − se simplifient) :

```
T = RC · ln[ (Vsat−V₂)(Vsat+V₁) / ((Vsat−V₁)(Vsat+V₂)) ]
```

**Les quatre facteurs.** Astuce : 1 + a = (1−a)·k.

```
Vsat − V₂ = Vsat[1 + a − (1−a)ρ] = Vsat(1−a)(k − ρ)
Vsat + V₁ = Vsat[1 + a + (1−a)ρ] = Vsat(1−a)(k + ρ)
Vsat − V₁ = Vsat[1 − a − (1−a)ρ] = Vsat(1−a)(1 − ρ)
Vsat + V₂ = Vsat[1 − a + (1−a)ρ] = Vsat(1−a)(1 + ρ)
```

**Simplification.** Le produit [Vsat(1−a)]² disparaît en haut et en bas, puis (x−y)(x+y) = x² − y² :

```
T = RC · ln[ (k−ρ)(k+ρ) / ((1−ρ)(1+ρ)) ] = RC · ln[ (k² − ρ²)/(1 − ρ²) ]
T = RC · ln[ ((1 + 2R₁/R₂)² − (Vref/Vsat)²) / (1 − (Vref/Vsat)²) ]

t₃ − t₂ = RC · ln[(k − ρ)/(1 − ρ)]        t₂ − t₁ = RC · ln[(k + ρ)/(1 + ρ)]
```

Condition d'oscillation : |V_ref| < Vsat, pour que les seuils restent dans la plage de saturation.

**Vérification numérique** (R₁ = R₂, Vsat = 12 V, V_ref = 3 V) : a = 1/2, k = 3, ρ = 0,25, V₁ = 7,5 V, V₂ = −4,5 V.

| Quantité | Calcul direct | Avec la formule |
|---|---|---|
| Vsat − V₂ | 12 + 4,5 = 16,5 | 12 · 0,5 · (3 − 0,25) = 16,5 |
| Vsat + V₁ | 12 + 7,5 = 19,5 | 12 · 0,5 · (3 + 0,25) = 19,5 |
| Vsat − V₁ | 12 − 7,5 = 4,5 | 12 · 0,5 · (1 − 0,25) = 4,5 |
| Vsat + V₂ | 12 − 4,5 = 7,5 | 12 · 0,5 · (1 + 0,25) = 7,5 |

T = RC · ln(8,9375/0,9375) = RC · ln 9,533 ≈ 2,255 RC.

Une simulation de l'équation du condensateur avec les deux seuils confirme les formules :

| R₁, R₂, V_ref | T formule | δ |
|---|---|---|
| 10, 10, 0 | 2,1972 RC | 0,5 |
| 10, 10, 3 V | 2,2548 RC | 0,5762 |
| 10, 22, −4 V | 1,3801 RC | 0,3767 |
| 47, 10, 0 | 4,6836 RC | 0,5 |

**Cas du cours V_ref = 0** : ρ = 0, V₁ = −V₂ = a·Vsat.

```
T = 2·RC·ln(1 + 2R₁/R₂)
t₃−t₂ = t₂−t₁ = RC·ln(1 + 2R₁/R₂)        δ = 0,5 (constant)
```

Le signal est un carré symétrique : la fréquence se règle, pas le rapport cyclique. Exemple : R = 100 kΩ, C = 1 µF, R₁ = R₂ donnent RC = 0,1 s et T = 2 × 0,1 × ln 3 ≈ 0,22 s (≈ 4,55 Hz), seuils ±6 V pour Vsat = 12 V.

Avec V_ref ≠ 0, δ s'écarte de 0,5 mais T change aussi : ce n'est pas un moyen de régler δ seul.

### 3.6 Régler δ : première modification avec deux diodes

Deux branches en parallèle remplacent R, séparées par deux diodes en sens opposés : D avec R, D' avec R'. (Sur le schéma manuscrit, la résistance du haut est écrite « R₁ » mais c'est bien R'.)

| Phase | Sortie | Condensateur | Diode passante | Résistance |
|---|---|---|---|---|
| t₂ < t < t₃ | Vsat⁺ | se charge | D | R |
| t₁ < t < t₂ | Vsat⁻ | se décharge | D' | R' |

**Calcul de t₂ − t₁** (V_ref = 0). Formule de 3.4 avec R → R' :

```
t₂ − t₁ = R'C · ln[ (−Vsat − R₁Vsat/(R₁+R₂)) / (−Vsat + R₁Vsat/(R₁+R₂)) ]
```

On multiplie haut et bas par (R₁+R₂) :

```
haut : −Vsat(R₁+R₂) − R₁Vsat = −Vsat(2R₁ + R₂)
bas  : −Vsat(R₁+R₂) + R₁Vsat = −Vsat·R₂
rapport = (2R₁+R₂)/R₂ = 1 + 2R₁/R₂
```

Résultats :

```
t₂ − t₁ = R'C · ln(1 + 2R₁/R₂)        t₃ − t₂ = R·C · ln(1 + 2R₁/R₂)
T = (R + R')·C · ln(1 + 2R₁/R₂)        δ = R/(R + R')
```

(Diodes idéales : avec des diodes réelles, la chute d'environ 0,6 V décale un peu les durées.)

Limite : régler δ en changeant R' change aussi T. Exemples : R' = R/3 donne δ = 0,75 ; R' = R donne 0,5 ; R' = 3R donne 0,25, avec T qui varie à chaque fois.

### 3.7 Solution finale : potentiomètre entre les deux diodes

Potentiomètre de résistance totale R, curseur K (0 < K < 1). Chemin de charge par D : K·R. Chemin de décharge par D' : (1−K)·R.

```
T = (K·R + (1−K)·R)·C·ln(1 + 2R₁/R₂) = R·C·ln(1 + 2R₁/R₂)
δ = K·R / (K·R + (1−K)·R) = K
```

La somme KR + (1−K)R vaut toujours R : T est constant, δ = K ne dépend d'aucune résistance. On règle δ avec le curseur sans changer T.

### 3.8 Application : variation de vitesse d'un moteur (MLI)

La sortie de l'astable (signal MLI) commande la grille d'un NMOS en série avec le moteur, alimenté sous +Vcc. Le NMOS « laisse passer + » : il conduit quand V_s = Vsat⁺ et se bloque quand V_s = Vsat⁻ (grille négative).

```
0 < t < T₀ :  U(t) = Vcc   (NMOS passant)          T₀ = δ·T = K·T
T₀ < t < T :  U(t) = 0 V   (NMOS bloqué)
```

La diode en parallèle sur le moteur est une diode de roue libre : le moteur est inductif, le courant ne peut pas s'arrêter instantanément, la diode lui offre un chemin à la coupure et protège le transistor.

Tension moyenne :

```
Vmoy = (1/T) ∫₀ᵀ U(t) dt = (1/T) ∫₀^T₀ Vcc dt = (T₀/T)·Vcc = δ·Vcc = K·Vcc
```

La deuxième intégrale (de T₀ à T) est nulle car U = 0. La vitesse est à peu près proportionnelle à Vmoy : variation linéaire en K, de 0 (K ≈ 0) à la vitesse maximale (K ≈ 1), sans changer T.

Condition pratique : T doit être petit devant la constante de temps mécanique du moteur. T ≈ 0,22 s ferait des à-coups. Avec R = 100 kΩ, C = 10 nF et R₁ = R₂ : T = RC·ln 3 ≈ 1,1 ms, soit ≈ 910 Hz.

### 3.9 Méthode à savoir refaire seul

1. Écrire e⁺ par le pont diviseur, puis V₁ et V₂.
2. Pour chaque phase, noter Vs, le seuil actif, la valeur de départ de Vc et la valeur visée.
3. Écrire Vc(t) avec la solution du premier ordre.
4. Écrire la condition de fin de phase (Vc = V₁ ou Vc = V₂) et résoudre avec un logarithme.
5. Additionner les deux durées pour T, diviser la durée à l'état haut par T pour δ.
6. Contrôler les signes : durées positives et |V_ref| < Vsat.

---

## 4. Calcul de e⁺ pour un nœud à trois résistances

Schéma : le point C (tension V_c) est relié par R au nœud N de l'entrée (+). Au nœud N : une résistance R vers la pile E (pôle − à la masse, donc le point relié à R est à +E), une résistance R vers la masse. Le courant dans l'entrée (+) est nul.

**Méthode 1, loi des nœuds (Millman).**

```
(Vc − e⁺)/R + (E − e⁺)/R + (0 − e⁺)/R = 0
Vc + E = 3·e⁺
e⁺ = (Vc + E)/3
```

Formulation du cours : e⁺ = (Vc/R + E/R + 0/R)/(3/R). Le terme 0/R vient de la branche vers la masse.

**Méthode 2, superposition.** Le circuit est linéaire : l'effet des deux sources est la somme des effets de chacune. Une source de tension éteinte est remplacée par un fil (0 V).

- **Vc seul (E éteinte).** La pile devient un fil : deux R vers la masse en parallèle, R // R = R/2. Pont diviseur : e⁺ = Vc · (R/2)/(R + R/2) = Vc/3.
- **E seule (Vc éteinte).** Le point C est relié à la masse : la résistance venant de C et celle vers la masse sont en parallèle, R/2. Même pont : e⁺ = E/3.
- **Somme :** e⁺ = (Vc + E)/3.

On peut traiter les diviseurs seuls parce que l'AOP ne prélève aucun courant.

Contrôle numérique (Vc = 6 V, E = 3 V) : 6/3 + 3/3 = 3 V. Loi des nœuds avec e⁺ = 3 V : 3/R + 0 − 3/R = 0.

Cas général (résistances différentes R₁ vers C, R₂ vers E, R₃ vers la masse) :

```
e⁺ = (Vc/R₁ + E/R₂) / (1/R₁ + 1/R₂ + 1/R₃)
```

---

## 5. La 2ᵉ partie du circuit : V_D = (2/3)(V_c + E)

Le même nœud alimente l'entrée (+) d'un AOP. L'entrée (−) est reliée à la masse par R et à la sortie V_D par R.

- e⁺ = (Vc + E)/3 (section 4).
- e⁻ = V_D · R/(R+R) = V_D/2 (pont diviseur, courant d'entrée nul).
- Réaction négative donc **régime linéaire** : e⁺ = e⁻.

```
V_D/2 = (Vc + E)/3
V_D = (2/3)·(Vc + E) = (2/3)·Vc + (2/3)·E
```

Montage : diviseur par 3 suivi d'un amplificateur non inverseur de gain 1 + R/R = 2.

**Remarque sur le cours.** Le résultat encadré est écrit « V_D = 2/3 Vc(t) + E ». D'après le calcul, le décalage est (2/3)·E, pas E.

**Valeurs 0,3 V et 0,6 V (hypothèse).** Si E = 0,9 V, E/3 = 0,3 V (niveau marqué sur l'axe du graphe) et 2E/3 = 0,6 V (annotation près du résultat). Ces deux valeurs sont cohérentes entre elles, mais E n'est pas confirmé (voir section 11).

**Graphe et conclusion du cours.** Vc(t) est proche de 0 avec des impulsions aux fronts. V_D reprend ces impulsions, réduites à 2/3 et décalées de 2E/3. Entre les impulsions, V_D = 2E/3 ≠ 0 : « générateur d'impulsion avec une valeur de stabilité (≠ 0) ».

| Étape | Résultat |
|---|---|
| Entrée (+) | e⁺ = (Vc + E)/3 |
| Entrée (−) | e⁻ = V_D/2 |
| Régime linéaire | e⁺ = e⁻ |
| Sortie | V_D = (2/3)(Vc + E) |
| Niveau de repos (Vc ≈ 0) | V_D = 2E/3 |

---

## 6. Pourquoi ce n'est pas un trigger non inverseur

Un trigger a une **réaction positive** : la sortie est ramenée sur l'entrée (+). Dans le montage de la section 5, V_D est ramené sur l'entrée (−) : réaction négative, donc amplificateur linéaire. Le nœud de l'entrée (+) ne reçoit que Vc, E et la masse.

Les schémas A (trigger non inverseur) et B (montage du cours) sont dans `triggers.png`, avec la réaction en rouge.

![Trigger non inverseur (A) et montage du cours (B)](triggers.png)

| | A. Trigger non inverseur | B. Montage du cours |
|---|---|---|
| Sortie ramenée sur | entrée (+), par R₂ | entrée (−), par R |
| Réaction | positive | négative |
| Entrée (−) | masse (ou V_ref) | diviseur R/R (sortie et masse) |
| Entrée (+) | signal e via R₁, et sortie via R₂ | Vc, E et masse (3 R) |
| Régime de l'AOP | saturé | linéaire (e⁺ = e⁻) |
| Sortie | seulement ±Vsat | V_D = (2/3)(Vc + E) |
| Seuils, hystérésis | deux seuils, oui | non |
| Fonction | comparateur | amplificateur |

Seuils du trigger non inverseur (entrée (−) à la masse) : e⁺ = (R₂·e + R₁·Vs)/(R₁+R₂) = 0 donne e = −(R₁/R₂)·Vs. La sortie passe à +Vsat quand e > +(R₁/R₂)·Vsat et à −Vsat quand e < −(R₁/R₂)·Vsat.

Test rapide : la sortie est-elle reliée à l'entrée (+) ? Oui, c'est un trigger. Y a-t-il un chemin de la sortie vers (−) sans chemin vers (+) ? Oui, c'est un amplificateur linéaire. Aucun des deux ? Comparateur simple en boucle ouverte.

---

## 7. Exercice 2 : monostable déclenché par un front

### Lecture du montage

| Étage | Structure | Rôle |
|---|---|---|
| AOP 1 | Ve → C₁ → (−), R en contre-réaction | dérivateur inverseur |
| AOP 2 | sortie AOP 1 via R, pile E via R, R de contre-réaction, tous sur (−) | sommateur inverseur |
| AOP 3 | (+) reliée à R₁ (masse) et à R₂ en série avec C₂ vers Vs | comparateur à réaction positive |

Polarité de la pile : flèche E vers le bas et « + » côté masse, donc le point haut de la pile est à −E. Ceci est cohérent avec le graphe de l'énoncé (V₀ positif).

Notations : τ = (R₁+R₂)·C₂ et K = R₁(Vsat⁺ − Vsat⁻)/(R₁+R₂).

**Q1.**

```
AOP 1 : i = C₁·dVe/dt  →  V₁ = −R·C₁·dVe/dt
AOP 2 : entrée (−) = masse virtuelle :  V₁/R + (−E)/R + e'₃/R = 0
e'₃(t) = −V₁ + E = E + R·C₁·dVe/dt
```

**Q2.** Ve constant : e'₃ = V₀ = E > 0. En continu, C₂ est un circuit ouvert, aucun courant dans R₂ et R₁, donc e⁺₃ = 0. ε = 0 − E < 0 donc **Vs = Vsat⁻**, e⁺₃ = 0.

**Q3.** Impulsion positive en t₀ (front montant de Ve) : e'₃ monte, ε reste négatif. La tension de C₂ est continue et Vs ne change pas : **Vs(t₀) = Vsat⁻, e⁺₃(t₀) = 0**. Sans effet.

**Q4.** Impulsion négative en t₁ (front descendant de Ve), assez grande pour que e'₃(t₁) < 0.

- a. ε = 0 − e'₃(t₁) > 0 : **Vs(t₁) = Vsat⁺**.
- b. Soit v_C la tension de C₂ (de la sortie vers l'entrée (+)). Maille : Vsat⁺ = v_C + (R₁+R₂)·C₂·dv_C/dt, avec e⁺₃ = R₁·C₂·dv_C/dt et v_C(t₁) = Vsat⁻ (C₂ était chargé à Vsat⁻).

```
v_C(t) = Vsat⁺ − (Vsat⁺ − Vsat⁻)·e^(−(t−t₁)/τ)
e⁺₃(t) = K · e^(−(t−t₁)/τ)
```

- c. Au saut de Vs, C₂ garde sa tension et R₂, R₁ forment un pont diviseur sur le saut : **e⁺₃(t₁) = K = R₁(Vsat⁺ − Vsat⁻)/(R₁+R₂)**. Pour que Vs reste à Vsat⁺ après l'impulsion (e'₃ revient à E), il faut K > E.
- d. Rebasculement quand e⁺₃ = e'₃ = E :

```
K·e^(−(t₂−t₁)/τ) = E
t₂ = t₁ + (R₁+R₂)·C₂·ln[ R₁(Vsat⁺ − Vsat⁻) / ((R₁+R₂)·E) ]
```

Vs(t₂) = Vsat⁻.

**Q5.** À t₂, Vs chute de Vsat⁺ − Vsat⁻ et e⁺₃ chute de K. Ensuite v_C relaxe vers Vsat⁻ avec la même constante τ.

```
e⁺₃(t₂) = E − K  (négatif)       (juste avant : E)
e⁺₃(t) = (E − K)·e^(−(t−t₂)/τ)
```

Pour t > t₂, ε = e⁺₃ − E < 0 : Vs reste à Vsat⁻, e⁺₃ tend vers 0, retour à l'état stable.

**Q6.** Courbes dans `ex2.png`, tracées par simulation (Vsat = ±12 V, E = 2 V, R₁ = R₂ donc K = 12 V). On trouve t₂ − t₁ = τ·ln 6 ≈ 1,79 τ, comme la formule. e'₃ reste à E avec une impulsion positive en t₀ et une négative (sous 0) en t₁. e⁺₃ vaut 0 jusqu'à t₁, saute à K, décroît jusqu'à E en t₂, saute à E − K, remonte vers 0. Vs est à Vsat⁻, passe à Vsat⁺ entre t₁ et t₂, puis revient à Vsat⁻.

![e'₃, e⁺₃ et Vs de l'exercice 2](ex2.png)

**Q7.** **Monostable déclenché sur le front descendant de Ve** : un état stable (Vsat⁻), une impulsion négative sur e'₃ le fait basculer, retour seul après T = t₂ − t₁ = τ·ln(K/E), l'impulsion positive est ignorée. AOP 1 dérive Ve, AOP 2 inverse et ajoute le niveau E (qui fixe l'état stable), AOP 3 avec R₁, R₂, C₂ est le monostable.

Cas symétrique : T = (R₁+R₂)·C₂·ln[ 2R₁·Vsat / ((R₁+R₂)·E) ].

---

## 8. Exercice 3 : modulateur de largeur d'impulsion

Données : E = 15 V, R = 10 kΩ, Rth variable de 0 à 10 kΩ, Ve triangulaire de ±7,5 V, T = 2 ms. L'AOP est un comparateur en boucle ouverte : Ve sur (+), V_ref sur (−).

**Q1.**

Tension de référence (pont diviseur, courant d'entrée nul) :

```
V_ref = E · Rth/(R + Rth) = 15 · Rth/(10 kΩ + Rth)       de 0 V (Rth = 0) à 7,5 V (Rth = 10 kΩ)
```

Condition : Ve > V_ref donne Vs = Vsat⁺, sinon Vsat⁻. Ve monte de −7,5 V à +7,5 V entre 0 et T/2 (pente p = 15/(T/2) = 30/T, soit 15 V/ms). Le temps entre le croisement du seuil et le sommet est (7,5 − V_ref)/p = (7,5 − V_ref)·T/30. Le triangle est symétrique, donc la durée au-dessus du seuil est le double :

```
W = (T/15)·(7,5 − V_ref) = (T/2)·(1 − V_ref/7,5)
```

Comme E = 15 V = 2 × 7,5 V : V_ref/7,5 = 2·Rth/(R+Rth), d'où 1 − 2Rth/(R+Rth) = (R − Rth)/(R + Rth).

```
W = (T/2) · (R − Rth)/(R + Rth)  =  1 ms · (10 kΩ − Rth)/(10 kΩ + Rth)
```

| Rth | V_ref | W |
|---|---|---|
| 0 | 0 V | T/2 = 1 ms |
| 5 kΩ | 5 V | 0,33 ms (vérifié : 1 ms × 5/15) |
| 10 kΩ | 7,5 V | 0 (le seuil est au sommet du triangle) |

**Q2.** Modulateur de largeur d'impulsion (MLI, PWM) : la fréquence est fixe (T = 2 ms, 500 Hz) et la largeur W se règle avec Rth. Rapport cyclique δ = W/T = (1/2)(R − Rth)/(R + Rth), de 0 à 50 % (V_ref ≥ 0 limite à 50 %). Applications : vitesse de moteur à courant continu, gradateur de lumière, alimentations à découpage.

---

## 9. Exercice du bistable : correction détaillée

**Énoncé.** Montage d'un bistable, AOP parfait : Ve sur l'entrée (−), entrée (+) reliée à R₁ (masse) et à R₂ (sortie). Ve est une impulsion positive puis une impulsion négative.

1. Expliquer le principe de fonctionnement du bistable et tracer Ve(t) et Vs(t).
2. Proposer un montage à base de bistable qui commande la marche et l'arrêt d'un moteur à courant continu. Expliquer.

### Question 1 : principe de fonctionnement

**a. Le seuil de l'entrée (+).** Le courant d'entrée de l'AOP est nul : R₂ et R₁ sont en série entre Vs et la masse, et forment un pont diviseur.

```
e⁺ = R₁/(R₁+R₂) · Vs = a·Vs           avec a = R₁/(R₁+R₂) < 1
```

e⁺ dépend de la sortie, qui vaut +Vsat ou −Vsat : deux seuils, +a·Vsat et −a·Vsat. C'est la réaction positive qui crée l'hystérésis.

**b. Deux états stables.** Au repos Ve = 0, qui est entre −a·Vsat et +a·Vsat.

- Si Vs = +Vsat : e⁺ = +a·Vsat > 0 = e⁻, ε > 0, la sortie reste à +Vsat. Stable.
- Si Vs = −Vsat : e⁺ = −a·Vsat < 0 = e⁻, ε < 0, la sortie reste à −Vsat. Stable.

Le montage a donc deux états stables, c'est un bistable. Il mémorise le dernier état imposé.

**c. Conditions de basculement.**

| Sortie actuelle | Seuil e⁺ | Basculement quand | Sortie après |
|---|---|---|---|
| +Vsat | +a·Vsat | Ve > +a·Vsat (impulsion positive) | −Vsat |
| −Vsat | −a·Vsat | Ve < −a·Vsat (impulsion négative) | +Vsat |

Montage « inverseur » car Ve est sur (−) : une impulsion positive fait passer la sortie à l'état bas.

**d. Déroulement avec les deux impulsions.** On suppose Vs = +Vsat à t = 0.

- **Avant t₁ :** Vs = +Vsat, e⁺ = +a·Vsat, Ve = 0 < e⁺.
- **En t₁, impulsion positive** d'amplitude supérieure à a·Vsat : Ve > e⁺, ε < 0, **Vs passe à −Vsat**. Le seuil devient e⁺ = −a·Vsat.
- **Fin de l'impulsion :** Ve revient à 0 > e⁺ = −a·Vsat, donc ε < 0 : **Vs reste à −Vsat**. C'est l'effet mémoire.
- **En t₂, impulsion négative** d'amplitude inférieure à −a·Vsat : Ve < e⁺ = −a·Vsat, ε > 0, **Vs repasse à +Vsat**, et le seuil redevient +a·Vsat.

**e. Remarques.**

- Si l'état initial est Vs = −Vsat, l'impulsion positive est sans effet et seule l'impulsion négative fait basculer.
- Une impulsion plus petite que a·Vsat en valeur absolue n'a aucun effet : la largeur d'hystérésis 2a·Vsat donne l'immunité au bruit.
- Deux impulsions de même signe d'affilée ne changent rien la deuxième fois.
- La largeur de l'état bas est t₂ − t₁, imposée par l'entrée : pas de temporisation interne, contrairement au monostable.

**f. Chronogrammes.** `wave.png` montre Ve(t), le seuil e⁺(t) = a·Vs (pointillé rouge) et Vs(t), pour R₁ = R₂ (a = 1/2), Vsat = ±12 V : seuils ±6 V, impulsions de ±9 V. La simulation du montage donne un basculement en t₁ et un en t₂, aux fronts d'entrée des impulsions.

![Ve, e⁺ et Vs du bistable](wave.png)

### Question 2 : commande marche / arrêt d'un moteur à courant continu

Schéma dans `motor.png`.

![Commande marche/arrêt d'un moteur par un bistable](motor.png)

**Constitution.**

- Un bistable à AOP (trigger inverseur de la question 1), alimenté en ±Vcc, donc Vs = ±Vsat.
- Un NMOS en série avec le moteur (source à la masse, moteur de +Vcc au drain). Une résistance Rg de grille est facultative.
- Une diode de roue libre en parallèle sur le moteur, cathode vers +Vcc.
- Deux boutons-poussoirs : **Marche** relie Ve à −Vcc par Rm, **Arrêt** relie Ve à +Vcc par Ra. Rp ramène Ve à 0 V quand aucun bouton n'est enfoncé (Rp grande devant Ra et Rm).

**Fonctionnement.**

| Action | Ve | Vs | NMOS | Moteur |
|---|---|---|---|---|
| Repos après Marche | 0 V (via Rp) | +Vsat (mémorisé) | passant | tourne |
| Appui sur **Arrêt** | ≈ +Vcc > +a·Vsat | passe à −Vsat | V_GS = −Vsat, bloqué | s'arrête |
| Repos après Arrêt | 0 V | −Vsat (mémorisé) | bloqué | arrêté |
| Appui sur **Marche** | ≈ −Vcc < −a·Vsat | passe à +Vsat | V_GS = +Vsat, passant | tourne |

- À l'appui sur Marche, Ve = −Vcc·Rp/(Rm+Rp) ≈ −Vcc, sous le seuil −a·Vsat : la sortie passe à +Vsat, le NMOS conduit et le moteur est alimenté sous Vcc.
- Quand on relâche le bouton, Ve revient à 0 par Rp. Ve est entre les deux seuils : la sortie reste à +Vsat (e⁺ = +a·Vsat > 0) et le moteur continue de tourner. C'est la mémoire du bistable, comme un relais à auto-maintien.
- À l'appui sur Arrêt, Ve ≈ +Vcc > +a·Vsat : la sortie passe à −Vsat, le NMOS se bloque et le moteur s'arrête. L'état est conservé après relâchement.
- Appuyer sur Marche quand le moteur tourne déjà ne change rien (idem pour Arrêt à l'arrêt). Les rebonds mécaniques des contacts sont donc sans effet : une fois basculé, les impulsions du même sens ne changent plus rien.
- La diode de roue libre protège le NMOS : le moteur étant inductif, le courant doit pouvoir continuer à circuler à la coupure.

**Exemple de valeurs.** Vcc = 12 V, Vsat ≈ ±11 V, R₁ = R₂ (a = 1/2, seuils ±5,5 V), Ra = Rm = 1 kΩ, Rp = 100 kΩ : Ve ≈ ±11,9 V à l'appui, bien au-delà des seuils.

**Points d'attention.**

- Le NMOS doit supporter V_GS = ±Vsat (souvent ±20 V max) et avoir un courant de drain et un V_th adaptés au moteur.
- Pour une alimentation simple (non symétrique), il faut polariser l'AOP autrement : le schéma proposé suppose ±Vcc.
- Variante : un bistable non inverseur (Ve sur (+)) inverserait le sens de chaque bouton.

---

## 10. Formulaire

| Grandeur | Expression |
|---|---|
| Dérivateur | Vs = −R·C·dVe/dt |
| Pont diviseur | e = V · R_bas/(R_haut + R_bas) |
| Seuils astable | V₁ = (R₁Vsat⁺ + R₂Vref)/(R₁+R₂), V₂ = (R₁Vsat⁻ + R₂Vref)/(R₁+R₂) |
| Condensateur | Vc(t) = Vs + (Vc(t₀) − Vs)·e^(−(t−t₀)/RC) |
| Durée à Vsat⁺ | t₃ − t₂ = RC·ln[(Vsat⁺−V₂)/(Vsat⁺−V₁)] |
| Durée à Vsat⁻ | t₂ − t₁ = RC·ln[(Vsat⁻−V₁)/(Vsat⁻−V₂)] |
| Période (symétrique) | T = RC·ln[((1+2R₁/R₂)² − ρ²)/(1 − ρ²)], ρ = Vref/Vsat |
| V_ref = 0 | T = 2RC·ln(1+2R₁/R₂), δ = 0,5 |
| Deux diodes | T = (R+R')C·ln(1+2R₁/R₂), δ = R/(R+R') |
| Potentiomètre | T = RC·ln(1+2R₁/R₂), δ = K |
| Moteur (MLI) | Vmoy = K·Vcc |
| Trigger inverseur | seuils ±a·Vsat, a = R₁/(R₁+R₂) |
| Trigger non inverseur | seuils ±(R₁/R₂)·Vsat |
| Nœud à 3 R égales | e⁺ = (V_a + V_b)/3 (la 3ᵉ à la masse) |
| Amplificateur non inverseur | gain 1 + R_f/R_g |
| Exercice 2 | e'₃ = E + R·C₁·dVe/dt, T = τ·ln(K/E), τ = (R₁+R₂)C₂ |
| Exercice 3 | V_ref = E·Rth/(R+Rth), W = (T/2)(R−Rth)/(R+Rth) |

---

## 11. Points à confirmer

- **Exercice 2, polarité de E.** J'ai lu la flèche E vers le bas et le « + » côté masse comme point haut à −E, donc V₀ = +E. Si ton cours lit la flèche autrement, le niveau stable devient −E et les réponses changent.
- **Section 5, valeur de E.** L'hypothèse E = 0,9 V (0,3 V et 0,6 V) vient de la cohérence des annotations, elle n'est pas confirmée. Le « +E » du résultat encadré semble devoir être « + (2/3)E ».
- **Section 1.** Brochage des AOP et sens des diodes lus sur un schéma manuscrit : à confirmer avec ton cours.
- **Exercice du bistable, question 2.** Montage proposé par moi (boutons Marche et Arrêt). Une correction officielle peut utiliser un autre dispositif, avec le même principe de mémoire.
