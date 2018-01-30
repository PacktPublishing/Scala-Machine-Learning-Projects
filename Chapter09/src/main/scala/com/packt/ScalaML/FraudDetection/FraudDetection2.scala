package com.packt.ScalaML.FraudDetection

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.h2o._
import _root_.hex.FrameSplitter
import water.Key
import water.fvec.Frame
import _root_.hex.deeplearning.DeepLearning
import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters
import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation
import java.io.File
import water.support.ModelSerializationSupport
import _root_.hex.{ ModelMetricsBinomial, ModelMetrics }
import org.apache.spark.h2o._
import scala.reflect.api.materializeTypeTag
import water.support.ModelSerializationSupport
import water.support.ModelMetricsSupport
import _root_.hex.deeplearning.DeepLearningModel
import vegas._
import vegas.sparkExt._
import org.apache.spark.sql.types._

object FraudDetection2 {
  case class r(precision: Double, recall: Double)
  case class r2(sensitivity: Double, specificity: Double)  
  case class r3(tp: Double, fp: Double, tn: Double, fn: Double, th: Double)
  
  def toCategorical(f: Frame, i: Int): Unit = {
    f.replace(i, f.vec(i).toCategoricalVec)
    f.update()
  }

  def confusionMat(mSEs: water.fvec.Frame, actualFrame: water.fvec.Frame, thresh: Double): Array[Array[Int]] = {
    val actualColumn = actualFrame.vec("Class");
    val l2_test = mSEs.anyVec();
    val result = Array.ofDim[Int](2, 2)
    var i = 0
    var ii, jj = 0
    for (i <- 0 until l2_test.length().toInt) {
      ii = if (l2_test.at(i) > thresh) 1 else 0;
      jj = actualColumn.at(i).toInt
      result(ii)(jj) = result(ii)(jj) + 1
    }
    result
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "C:/Users/admin-karim/Downloads/tmp/")
      .appName("Fraud Detection")
      .getOrCreate()

    implicit val sqlContext = spark.sqlContext
    import sqlContext.implicits._
    val h2oContext = H2OContext.getOrCreate(spark)
    import h2oContext._
    import h2oContext.implicits._

    val inputCSV = "data/creditcard.csv"
    val transactions = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", true).load(inputCSV)
    transactions.select("Time", "V1", "v2","V26", "V27",  "Amount", "Class").show(10)  

    val distribution = transactions.groupBy("Class").count.collect

    Vegas("Class Distribution").withData(distribution.map(r => Map("class" -> r(0), "count" -> r(1)))).encodeX("class", Nom).encodeY("count", Quant).mark(Bar).show

    val daysUDf = udf((s: Double) => if (s > 3600 * 24) "day2" else "day1")
    val t1 = transactions.withColumn("day", daysUDf(col("Time")))
    val dayDist = t1.groupBy("day").count.collect

    Vegas("Day Distribution").withData(dayDist.map(r => Map("day" -> r(0), "count" -> r(1)))).encodeX("day", Nom).encodeY("count", Quant).mark(Bar).show

    val dayTimeUDf = udf((day: String, t: Double) => if (day == "day2") t - 86400 else t)
    val t2 = t1.withColumn("dayTime", dayTimeUDf(col("day"), col("Time")))
    t2.describe("dayTime").show()

    val d1 = t2.filter($"day" === "day1")
    val d2 = t2.filter($"day" === "day2")
    val quantiles1 = d1.stat.approxQuantile("dayTime", Array(0.25, 0.5, 0.75), 0)
    val quantiles2 = d2.stat.approxQuantile("dayTime", Array(0.25, 0.5, 0.75), 0)
    val bagsUDf = udf((t: Double) => if (t <= (quantiles1(0) + quantiles2(0)) / 2) "gr1" else if (t <= (quantiles1(1) + quantiles2(1)) / 2) "gr2" else if (t <= (quantiles1(2) + quantiles2(2)) / 2) "gr3" else "gr4")
    val t3 = t2.drop(col("Time")).withColumn("Time", bagsUDf(col("dayTime")))

    val grDist = t3.groupBy("Time", "class").count.collect
    val grDistByClass = grDist.groupBy(_(1))

    Vegas("gr Distribution")
      .withData(grDistByClass.get(0)
          .get.map(r => Map("Time" -> r(0), "count" -> r(2))))
          .encodeX("Time", Nom)
          .encodeY("count", Quant)
          .mark(Bar)
          .show

    Vegas("gr Distribution").withData(grDistByClass.get(1).get.map(r => Map("Time" -> r(0), "count" -> r(2)))).encodeX("Time", Nom).encodeY("count", Quant).mark(Bar).show

    val c0Amount = t3.filter($"Class" === "0").select("Amount")
    val c1Amount = t3.filter($"Class" === "1").select("Amount")
    
    println(c0Amount.stat.approxQuantile("Amount", Array(0.25, 0.5, 0.75), 0).mkString(","))
    //gives q1,median,q3
    println(c0Amount.stat.approxQuantile("Amount", Array(0.25, 0.5, 0.75), 0).mkString(","))
    //gives q1,median,q3

    Vegas("Amounts for class 0")
        .withDataFrame(c0Amount)
        .mark(Bar)
        .encodeX("Amount", Quantitative, bin = Bin(50.0))
        .encodeY(field = "*", Quantitative, aggregate = AggOps.Count)
        .show

    Vegas("Amounts for class 1")
        .withDataFrame(c1Amount)
        .mark(Bar)
        .encodeX("Amount", Quantitative, bin = Bin(50.0))
        .encodeY(field = "*", Quantitative, aggregate = AggOps.Count)
        .show

    val t4 = t3.drop("day").drop("dayTime")

    val creditcard_hf: H2OFrame = h2oContext.asH2OFrame(t4.orderBy(rand()))

    val sf = new FrameSplitter(creditcard_hf, Array(.4, .4), Array("train_unsupervised", "train_supervised", "test").map(Key.make[Frame](_)), null)
    water.H2O.submitTask(sf)
    val splits = sf.getResult
    val (train_unsupervised, train_supervised, test) = (splits(0), splits(1), splits(2))

    toCategorical(train_unsupervised, 30)
    toCategorical(train_supervised, 30)
    toCategorical(test, 30)

    val response = "Class"
    val features = train_unsupervised.names.filterNot(_ == response)
    var dlParams = new DeepLearningParameters()
    dlParams._ignored_columns = Array(response)
    dlParams._train = train_unsupervised._key
    dlParams._autoencoder = true
    dlParams._reproducible = true
    dlParams._ignore_const_cols = false
    dlParams._seed = 42
    dlParams._hidden = Array[Int](10, 2, 10)
    dlParams._epochs = 100
    dlParams._activation = Activation.Tanh
    dlParams._force_load_balance = false
    var dl = new DeepLearning(dlParams)
    val model_nn = dl.trainModel.get

    val uri = new File(new File(inputCSV).getParentFile, "model_nn.bin").toURI
    ModelSerializationSupport.exportH2OModel(model_nn, uri)

    val model: DeepLearningModel = ModelSerializationSupport.loadH2OModel(uri)
    println(model)

    val test_autoenc = model_nn.scoreAutoEncoder(test, Key.make(), false)

    var train_features = model_nn.scoreDeepFeatures(train_unsupervised, 1)
    train_features.add("Class", train_unsupervised.vec("Class"))

    train_features.setNames(train_features.names.map(_.replaceAll("[.]", "-")))
    train_features._key = Key.make()
    water.DKV.put(train_features)
    val tfDataFrame = asDataFrame(train_features)
    
    Vegas("Compressed")
        .withDataFrame(tfDataFrame)
        .mark(Point)
        .encodeX("DF-L2-C1", Quantitative)
        .encodeY("DF-L2-C2", Quantitative)
        .encodeColor(field = "Class", dataType = Nominal)
        .show

    train_features = model_nn.scoreDeepFeatures(train_unsupervised, 2)
    train_features._key = Key.make()
    train_features.add("Class", train_unsupervised.vec("Class"))
    water.DKV.put(train_features)

    val features_dim = train_features.names.filterNot(_ == response)
    val train_features_H2O = asH2OFrame(train_features)

    dlParams = new DeepLearningParameters()
    dlParams._ignored_columns = Array(response)
    dlParams._train = train_features_H2O
    dlParams._autoencoder = true
    dlParams._reproducible = true
    dlParams._ignore_const_cols = false
    dlParams._seed = 42
    dlParams._hidden = Array[Int](10, 2, 10)
    dlParams._epochs = 100
    dlParams._activation = Activation.Tanh
    dlParams._force_load_balance = false
    dl = new DeepLearning(dlParams)
    val model_nn_dim = dl.trainModel.get
 
    ModelSerializationSupport.exportH2OModel(model_nn_dim, new File(new File(inputCSV).getParentFile, "model_nn_dim.bin").toURI)

    val test_dim = model_nn.scoreDeepFeatures(test, 2)
    val test_dim_score = model_nn_dim.scoreAutoEncoder(test_dim, Key.make(), false)
    println(test_dim_score)
    val result = confusionMat(test_dim_score, test, test_dim_score.anyVec.mean)
    println(result.deep.mkString("\n"))

    test_dim_score.add("Class", test.vec("Class"))
    val testDF = asDataFrame(test_dim_score).rdd.zipWithIndex.map(r => Row.fromSeq(r._1.toSeq :+ r._2))

    val schema = StructType(Array
                  (StructField("Reconstruction-MSE", DoubleType, nullable = false), 
                  StructField("Class", ByteType, nullable = false), 
                  StructField("idRow", LongType, nullable = false))
                      )
    val dffd = spark.createDataFrame(testDF, schema)
    dffd.show()
    
    Vegas("Reduced Test", width = 800, height = 600)
        .withDataFrame(dffd).mark(Point)
        .encodeX("idRow", Quantitative)
        .encodeY("Reconstruction-MSE", Quantitative)
        .encodeColor(field = "Class", dataType = Nominal)
        .show

    toCategorical(train_supervised, 29)

    val train_supervised_H2O = asH2OFrame(train_supervised)
    dlParams = new DeepLearningParameters()
    dlParams._pretrained_autoencoder = model_nn._key
    dlParams._train = train_supervised_H2O
    dlParams._reproducible = true
    dlParams._ignore_const_cols = false
    dlParams._seed = 42
    dlParams._hidden = Array[Int](10, 2, 10)
    dlParams._epochs = 100
    dlParams._activation = Activation.Tanh
    dlParams._response_column = "Class"
    dlParams._balance_classes = true
    dl = new DeepLearning(dlParams)
    
    val model_nn_2 = dl.trainModel.get
    val predictions = model_nn_2.score(test, "predict")
    test.add("predict", predictions.vec("predict"))
    asDataFrame(test).groupBy("Class", "predict").count.show //print

    h2oContext.stop(stopSparkContext = true)
    spark.stop()
  }
}

