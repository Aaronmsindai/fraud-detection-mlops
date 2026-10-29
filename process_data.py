from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# 1. Initialize Spark Session
print("Starting Spark Session...")
spark = SparkSession.builder \
    .appName("FraudDetection") \
    .getOrCreate()

# 2. Load the CSV into a Spark DataFrame
print("Loading 1 million transactions...")
df = spark.read.csv("transactions.csv", header=True, inferSchema=True)

# 3. Data Exploration
print(f"Total rows: {df.count()}")
print(f"Fraud cases: {df.filter(col('is_fraud') == 1).count()}")

# 4. Feature Engineering
feature_cols = ['amount', 'hour_of_day', 'user_age', 'is_international']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_features = assembler.transform(df)

# 5. Train/Test Split
train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=42)

# 6. Train a Random Forest Model
print("Training Random Forest model...")
rf = RandomForestClassifier(labelCol="is_fraud", featuresCol="features", numTrees=50)
model = rf.fit(train_data)

# 7. Make Predictions
print("Evaluating model...")
predictions = model.transform(test_data)

# 8. Evaluate with AUC
evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"Model AUC: {auc:.4f}")

# 9. Stop Spark
spark.stop()
print("Done!")
