# Lab-9
Python 
import numpy as np

# ==========================================
# Étape 1 : Préparation de l'environnement
# ==========================================

# Task 1.1 & 1.2 - Vérification de la version de NumPy
print("--- Étape 1 : Vérification ---")
print("Version de NumPy installée :", np.__version__)
print("\n" + "="*50 + "\n")


# ==========================================
# Étape 2 : Création de tableaux NumPy
# ==========================================
print("--- Étape 2 : Création de tableaux ---")

# Task 2.1 - Tableaux 1D et 2D avec np.array
tableau_1d = np.array([1, 2, 3, 4, 5])
print("Tableau 1D :", tableau_1d)

tableau_2d = np.array([[1, 2, 3], 
                       [4, 5, 6]])
print("Tableau 2D :\n", tableau_2d)

# Task 2.2 - Générateurs de tableaux
zeros = np.zeros((3, 3))
print("Zeros (3x3) :\n", zeros)

ones = np.ones((3, 2))
print("Ones (3x2) :\n", ones)

progression = np.arange(0, 10, 2)
print("np.arange(0, 10, 2) :", progression)

linspace = np.linspace(0, 1, 5)
print("np.linspace(0, 1, 5) :", linspace)

# Insight : Vectorisation des opérations
print("Vectorisation (tableau_1d + 5) :", tableau_1d + 5)
print("\n" + "="*50 + "\n")


# ==========================================
# Étape 3 : Inspection et métadonnées
# ==========================================
print("--- Étape 3 : Inspection et métadonnées ---")

# Task 3.1 - Affichage des métadonnées
def afficher_metadonnes(nom, tableau):
    print(f"--- Métadonnées pour : {nom} ---")
    print("Contenu :\n", tableau)
    print("dtype :", tableau.dtype)   # Type des éléments
    print("ndim  :", tableau.ndim)    # Nombre de dimensions
    print("shape :", tableau.shape)   # Tuple de dimensions
    print("size  :", tableau.size)    # Nombre total d'éléments
    print()

afficher_metadonnes("tableau_1d", tableau_1d)
afficher_metadonnes("tableau_2d", tableau_2d)
afficher_metadonnes("zeros", zeros)

# Task 3.2 - Explications ciblées
print("--- Explications (Task 3.2) ---")
# 1. np.zeros renvoie des float64 par défaut
print("1. Type par défaut de np.zeros :", zeros.dtype)

# 2. Promotion de type (Type Promotion)
tableau_promu = np.array([1, 2, 3.0])
print("2. Type de np.array([1, 2, 3.0]) :", tableau_promu.dtype) 

# 3. Différence entre type() et dtype
print("3. type(tableau_1d) :", type(tableau_1d)) # Toujours <class 'numpy.ndarray'>
print("   tableau_1d.dtype :", tableau_1d.dtype)  # Type interne des éléments (ex: int64)
print("\n" + "="*50 + "\n")


# ==========================================
# Étape 4 : Reshape (Modifier la forme)
# ==========================================
print("--- Étape 4 : Reshape ---")

# Task 4.1 - Restructurer un vecteur
vecteur = np.arange(1, 10) # 1 à 9
print("Vecteur initial :", vecteur)

matrice = vecteur.reshape((3, 3))
print("Matrice 3x3 :\n", matrice)

print("Shape de la matrice :", matrice.shape)
print("Size de la matrice  :", matrice.size)

# Task 4.2 - Dimension déduite automatiquement (-1)
matrice_auto = vecteur.reshape((3, -1))
print("Shape calculée automatiquement avec -1 :", matrice_auto.shape)
# Note : NumPy calcule la dimension manquante (-1) selon le nombre total d'éléments.
# Si le produit des dimensions ne correspond pas à size, NumPy déclenche une ValueError.
print("\n" + "="*50 + "\n")


# ==========================================
# Étape 5 : Exploration supplémentaire
# ==========================================
print("--- Étape 5 : Exploration supplémentaire ---")

# Task 5.1 - Types et formats
tab_float32 = np.array([1, 2, 3], dtype=np.float32)
tab_int32 = tab_float32.astype(np.int32)
print("Tableau float32 :", tab_float32, "| Mémoire utilisée :", tab_float32.nbytes, "octets")
print("Tableau int32   :", tab_int32,   "| Mémoire utilisée :", tab_int32.nbytes, "octets")

# Task 5.2 - Tableaux utilitaires
identite = np.eye(4)
print("Matrice Identité (4x4) :\n", identite)

tableau_constant = np.full((2, 2), 7)
print("Tableau constant (2x2 rempli de 7) :\n", tableau_constant)

test_forme = np.arange(12).reshape((4, 3))
print("np.arange(12).reshape((4, 3)) :\n", test_forme)

# Task 5.3 - Opérations vectorisées
print("Multiplication vectorisée (matrice * 10) :\n", matrice * 10)
print("Racine carrée vectorisée (np.sqrt(vecteur)) :\n", np.sqrt(vecteur))

# Intérêt de la vectorisation :
# La vectorisation permet d'effectuer des calculs directement sur des tableaux entiers en C 
# sans utiliser de boucles Python (for), ce qui offre une exécution très rapide et un code concis.
