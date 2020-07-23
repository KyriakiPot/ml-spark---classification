from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

data = spark.read.format("csv").load("hdfs://master:9000/user/user/input/HIGGS.csv.gz")

df1 = data.select(data._c0,data._c1.cast("double"),data._c2.cast("double"),data._c3.cast("double"),data._c4.cast("double"),data._c5.cast("double"),
data._c6.cast("double"),data._c7.cast("double"),data._c8.cast("double"),data._c9.cast("double"),data._c10.cast("double"),data._c11.cast("double"),data._c12.cast("double"),
data._c13.cast("double"),data._c14.cast("double"),data._c15.cast("double"),data._c16.cast("double"),data._c17.cast("double"),data._c18.cast("double"),data._c19.cast("double"),
data._c20.cast("double"),data._c21.cast("double"),data._c22.cast("double"),data._c23.cast("double"),data._c24.cast("double"),data._c25.cast("double"),data._c26.cast("double"),
data._c27.cast("double"),data._c28.cast("double"))

assembler = VectorAssembler(inputCols = ["_c1","_c2","_c3","_c4","_c5","_c6","_c7","_c8","_c9","_c10","_c11","_c12","_c13","_c14","_c15","_c16","_c17","_c18","_c19","_c20","_c21","_c22","_c23","_c24","_c25","_c26","_c27","_c28"],outputCol = ('features'))
output = assembler.transform(df1)
df = output.drop("_c1","_c2","_c3","_c4","_c5","_c6","_c7","_c8","_c9","_c10","_c11","_c12","_c13","_c14","_c15","_c16","_c17","_c18","_c19","_c20","_c21","_c22","_c23","_c24","_c25","_c26","_c27","_c28")

labelIndexer = StringIndexer(inputCol="_c0", outputCol="indexedLabel").fit(df)

(trainingData, testData) = df.randomSplit([0.7, 0.3])
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features", maxDepth = 16,maxBins = 64,maxMemoryInMB=512, cacheNodeIds=True, checkpointInterval=15, impurity="entropy")

pipeline = Pipeline(stages=[labelIndexer,dt])

model = pipeline.fit(trainingData)

predictions = model.transform(testData)

predictions.show(5)

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType

preds_and_labels = predictions.select(predictions.indexedLabel.cast("float"),predictions.prediction).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','indexedLabel'])
pl = preds_and_labels.rdd.map(tuple)
metrics = MulticlassMetrics(pl)
print(metrics.confusionMatrix().toArray())

################################################################
evaluator = MulticlassClassificationEvaluator( labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % (metrics.accuracy))

treeModel = model.stages[1]
# summary only
print(treeModel)



data = spark.read.format("csv").load("hdfs://master:9000/user/user/input/HIGGS.csv.gz")
data = spark.read.format("csv").load("sinput/HIGGS.csv.gz")