from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, count, sum as _sum, length

# Configurer Spark pour moins de logs
spark = SparkSession.builder \
    .appName("PageRank") \
    .config("spark.ui.showConsoleProgress", "false") \
    .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=file:/usr/lib/spark/conf/log4j.properties") \
    .getOrCreate()

# Réduire le niveau des logs
spark.sparkContext.setLogLevel("WARN")

# Lire edges.csv depuis GCS, sans en-tête, nommer les colonnes
edges = spark.read.csv("gs://pagerank-wikipedia/edges/edges.csv", header=False) \
            .toDF("src", "dst")

# Nettoyer les URLs (enlever les < et > si présents)
edges = edges.withColumn("src", col("src").substr(lit(2), length(col("src")) - 2)) \
             .withColumn("dst", col("dst").substr(lit(2), length(col("dst")) - 2))

# Afficher un échantillon pour vérifier
print("=== Échantillon des edges (premières 5 lignes) ===")
edges.show(5, truncate=False)

# Initialiser ranks à 1.0 pour chaque page unique
pages = edges.select("src").union(edges.select("dst")).distinct()
ranks = pages.withColumn("rank", lit(1.0))

print(f"=== Nombre total de pages uniques: {pages.count()} ===")

# Boucle PageRank simple (10 itérations)
for i in range(10):
    print(f"=== Itération {i+1}/10 ===")
    
    # Calculer le nombre de liens sortants pour chaque source
    out_degree = edges.groupBy("src").agg(count("dst").alias("out_links"))
    
    # Calculer les contributions
    contribs = edges.join(ranks, "src") \
                    .join(out_degree, "src") \
                    .select("dst", 
                           (col("rank") / col("out_links")).alias("contrib"))
    
    # Calculer les nouveaux ranks (somme des contributions + 0.15/N pour le damping factor)
    N = ranks.count()
    new_ranks = contribs.groupBy("dst") \
                        .agg(_sum("contrib").alias("sum_contrib")) \
                        .withColumn("rank", 
                                   col("sum_contrib") * 0.85 + 0.15/N) \
                        .drop("sum_contrib") \
                        .withColumnRenamed("dst", "src")
    
    # Mettre à jour les ranks pour l'itération suivante
    ranks = new_ranks
    
    # Afficher quelques résultats intermédiaires
    if i < 2 or i == 9:  # Afficher seulement au début et à la fin
        print(f"Top 3 pages à l'itération {i+1}:")
        ranks.orderBy(col("rank").desc()).show(3, truncate=False)

# Afficher les résultats finaux triés
print("\n=== RANKS FINAUX (triés par score descendant) ===")
final_ranks = ranks.orderBy(col("rank").desc())

print("Top 10 pages:")
final_ranks.show(10, truncate=False)

print(f"\n=== CALCUL TERMINÉ ===")
print(f"Nombre total de pages rankées: {final_ranks.count()}")

# Sauvegarder les résultats
final_ranks.write \
    .mode("overwrite") \
    .csv("gs://pagerank-wikipedia/output/pagerank_results", header=True)

print("Résultats sauvegardés dans: gs://pagerank-wikipedia/output/pagerank_results")
