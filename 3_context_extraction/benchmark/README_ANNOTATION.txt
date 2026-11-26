BENCHMARK DE DETECTION D'OBJETS

1. Annoter l'enfant avec la classe : child
2. Annoter chaque objet pertinent avec une boîte distincte.
3. Utiliser des noms en minuscules avec underscore.
4. Utiliser le même nom pour le même type d'objet.
5. Préférer une classe précise : cabinet plutôt que furniture.
6. Annoter tous les objets appartenant aux classes retenues,
   même lorsqu'ils sont éloignés de l'enfant.
7. Pour un objet partiellement caché, encadrer la partie visible.
8. Ne pas consulter les prédictions des modèles avant l'annotation.
9. Ne pas appliquer d'augmentation aux images.
10. Exporter les annotations au format COCO JSON.

La classe child sera utilisée pour l'évaluation de la proximité,
mais sera exclue de la comparaison des détecteurs d'objets.