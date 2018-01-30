package com.packt.ScalaML.BitCoin

import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.classification.{ GBTClassificationModel, GBTClassifier, RandomForestClassificationModel, RandomForestClassifier }
import org.apache.spark.ml.evaluation.{ BinaryClassificationEvaluator, MulticlassClassificationEvaluator }
import org.apache.spark.ml.feature.{ IndexToString, StringIndexer, VectorAssembler, VectorIndexer }
import org.apache.spark.ml.tuning.{ CrossValidator, ParamGridBuilder }
import org.apache.spark.sql.types.{ DoubleType, IntegerType, StructField, StructType }
import org.apache.spark.sql.SparkSession

/**
 * Created by Md. Rezaul Karim on 24.12.17.
 * rezaul.karim.fit@gmail.com
 */

object TrainGradientBoostedTree {
  def main(args: Array[String]): Unit = {
    val maxBins = Seq(5, 7, 9)
    val numFolds = 10
    val maxIter: Seq[Int] = Seq(10)
    val maxDepth: Seq[Int] = Seq(20)
    val rootDir = "output/" // "data/"

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .config("spark.local.dir", "data/")
      .appName("Bitcoin Preprocessing")
      .getOrCreate()

    val xSchema = StructType(Array(
      StructField("t0", DoubleType, true),
      StructField("t1", DoubleType, true),
      StructField("t2", DoubleType, true),
      StructField("t3", DoubleType, true),
      StructField("t4", DoubleType, true),
      StructField("t5", DoubleType, true),
      StructField("t6", DoubleType, true),
      StructField("t7", DoubleType, true),
      StructField("t8", DoubleType, true),
      StructField("t9", DoubleType, true),
      StructField("t10", DoubleType, true),
      StructField("t11", DoubleType, true),
      StructField("t12", DoubleType, true),
      StructField("t13", DoubleType, true),
      StructField("t14", DoubleType, true),
      StructField("t15", DoubleType, true),
      StructField("t16", DoubleType, true),
      StructField("t17", DoubleType, true),
      StructField("t18", DoubleType, true),
      StructField("t19", DoubleType, true),
      StructField("t20", DoubleType, true),
      StructField("t21", DoubleType, true)))

    val ySchema = StructType(Array(StructField("y", DoubleType, true)))

    val x = spark.read.format("csv").schema(xSchema).load(rootDir + "scala_test_x.csv")
    val y_tmp = spark.read.format("csv").schema(ySchema).load(rootDir + "scala_test_y.csv")
    
    import spark.implicits._
    val y = y_tmp.withColumn("y", 'y.cast(IntegerType))
    import org.apache.spark.sql.functions._

    //joining 2 separate datasets produced by python in single Spark dataframe
    val x_id = x.withColumn("id", monotonically_increasing_id())
    val y_id = y.withColumn("id", monotonically_increasing_id())
    val data = x_id.join(y_id, "id")

    val featureAssembler = new VectorAssembler()
      .setInputCols(Array("t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12", "t13", "t14", "t15", "t16", "t17", "t18", "t19", "t20", "t21"))
      .setOutputCol("features")

    val encodeLabel = udf[Double, String] { case "1" => 1.0 case "0" => 0.0 }

    val dataWithLabels = data.withColumn("label", encodeLabel(data("y")))

    //123 is seed number to get same datasplit so we can tune params
    val Array(trainingData, testData) = dataWithLabels.randomSplit(Array(0.75, 0.25), 123)

    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setSeed(123)

    val pipeline = new Pipeline()
      .setStages(Array(featureAssembler, gbt))

    // ***********************************************************
    println("Preparing K-fold Cross Validation and Grid Search")
    // ***********************************************************
    val paramGrid = new ParamGridBuilder()
      .addGrid(gbt.maxIter, maxIter)
      .addGrid(gbt.maxDepth, maxDepth)
      .addGrid(gbt.maxBins, maxBins)
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)
      .setSeed(123)

    // ************************************************************
    println("Training model with GradientBoostedTrees algorithm")
    // ************************************************************

    // Train model. This also runs the indexers.
    val cvModel = cv.fit(trainingData)
    cvModel.save(rootDir + "cvGBT_22_binary_classes_" + System.nanoTime() / 1000000 + ".model")

    println("Evaluating model on train and test data and calculating RMSE")
    // **********************************************************************

    // Make predictions.
    val predictions = cvModel.transform(testData)

    // Select (prediction, true label) and compute test error.
    val rocEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")
    val roc = rocEvaluator.evaluate(predictions)

    val prEvaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderPR")

    val pr = prEvaluator.evaluate(predictions)

    val gbtModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    gbtModel.save(rootDir + "__cv__gbt_22_binary_classes_" + System.nanoTime() / 1000000 + ".model")

    println("Area under ROC curve = " + roc)
    println("Area under PR curve= " + pr)

    println(predictions.select().show(1))
    spark.stop()
  }
}
