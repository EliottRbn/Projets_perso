# Projet personnel de reconnaissance faciale 

Ce github a pour objectif de regrouper l'ensemble des travaux que j'ai effectués dans le cadre de ce projet personnel. 

Ce projet a pour objectif de créer un algorithme de reconnaissance faciale sans l'utilisation des libraires standards de Python. L'idée étant de comprendre comment fonctionne la reconnaissance faciale ainsi que les algorithmes sous-jacents. C'est un projet purement personnel auquel j'y consacre du temps lorsque ceci est possible. 

La méthode utilisée jusqu'à présent pour la reconnaissance faciale est la méthode des SVM. Plusieurs versions ont été réalisées, en passant par des SVM à marges souples et dures simple, avec une base de données faite à la main. Les algorithmes ont ensuite évolués, je me suis tourné vers des bases de données plus robustes, et me sers actuellement de l'algorithme de résolution SMO. Plusieurs noyaux ont été testés et une phase de détermination des meilleurs hyper-paramètres est en cours. J'ai également implémenté trois descripteurs, les descripteurs Haar, HOG et LBP. Des tests de performances avec différentes métriques ont été réalisés avec ces descripteurs et leurs combinaisons. Malheureusement, bien que performants sur des données tests, il semble que je sois confronté à de l'over-fitting, rendant les tests caméra (via cv2) peu concluants.

Pour la suite, je vais essayer de trouver comment résoudre ce problème d'over-fitting, voir pour me tourner vers d'autres descripteurs ou d'autres bases de données, de sorte à ce que l'algorithme soit suffisamment robuste pour que les tests caméra soient concluants. 

Lorsque j'estimerai avoir suffisamment travaillé sur les SVM, je me tournerai sur les réseaux de neurones. Pour ce faire, je me baserai sur la base d'un programme réalisé lors d'un projet de mathématiques en troisième année à l'IPSA. 

De nombreuses modifications sont encore à faire, et le projet est en constante évolution. De nombreux tests unitaires restent encore à faire pour les différents descripteurs, partie sur laquelle je ne pense pas avoir suffisamment insistée jusqu'alors. Toutefois, ce dépôt github vous permettra de voir les avancées de ce dernier.
