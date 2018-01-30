package com.packt.ScalaML.FraudDetection

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
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
import org.apache.spark.h2o._
import scala.reflect.api.materializeTypeTag

object H2OCreditFraud {
  def toCategorical(f: Frame, i: Int): Unit = {
    f.replace(i, f.vec(i).toCategoricalVec)
    f.update()
  }

  def confusionMatrix(l2_frame_test: water.fvec.Frame, test: water.fvec.Frame, thresh: Double): Array[Array[Int]] = {
    val l2_test_dim = test.vec("Class")
    val l2_test = l2_frame_test.anyVec()
    val result = Array.ofDim[Int](2, 2)
    var (i, j, k) = (0, 0, 0)
    for (i <- 0 until l2_test.length().toInt) {
      j = if (l2_test.at(i) > thresh) 1 else 0
      k = l2_test_dim.at(i).toInt
      result(j)(k) = result(j)(k) + 1
    }
    result foreach { row => row foreach { e => print(e + " ") }; println }
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

    val path = "data/creditcard.csv"
    val transactions = spark.read.format("csv").option("header", "true").option("inferSchema", true).load(path)
    val daysUDf = udf((s: Double) => if (s > 3600 * 24) "day2" else "day1")
    val t1 = transactions.withColumn("day", daysUDf(col("Time")))
    t1.groupBy("day").count.show
    val dayTimeUDf = udf((day: String, t: Double) => if (day == "day2") t - 86400 else t)
    val t2 = t1.withColumn("dayTime", dayTimeUDf(col("day"), col("Time")))
    t2.describe("dayTime").show()

    val d1 = t2.filter($"day" === "day1")
    val d2 = t2.filter($"day" === "day2")
    val quantiles1 = d1.stat.approxQuantile("dayTime", Array(0.25, 0.5, 0.75), 0)
    val quantiles2 = d2.stat.approxQuantile("dayTime", Array(0.25, 0.5, 0.75), 0)
    
    val bagsUDf = udf((t: Double) => if (t <= (quantiles1(0) + quantiles2(0)) / 2) "gr1" else if (t <= (quantiles1(1) + quantiles2(1)) / 2) "gr2" else if (t <= (quantiles1(2) + quantiles2(2)) / 2) "gr3" else "gr4")
    val t3 = t2.drop(col("Time")).withColumn("Time", bagsUDf(col("dayTime")))
    t3.groupBy("day", "Time").count.show
    t3.filter($"Class" === "0").stat.approxQuantile("Amount", Array(0.25, 0.5, 0.75), 0)
    t3.filter($"Class" === "1").stat.approxQuantile("Amount", Array(0.25, 0.5, 0.75), 0)
    val t4 = t3.drop(col("day")).drop("dayTime")
    val creditcard_hf: H2OFrame = h2oContext.asH2OFrame(t4)

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

    ModelSerializationSupport.exportH2OModel(model_nn, new File("C:/Users/admin-karim/Downloads/tmp/model/kutta.bin").toURI)
    //val model:DeepLearningModel = ModelSerializationSupport.loadH2OModel(new File("/home/acer/model_nn.bin").toURI)
    val test_autoenc = model_nn.scoreAutoEncoder(test, Key.make(), false)

    var train_features = model_nn.scoreDeepFeatures(train_unsupervised, 1)
    train_features.add("Class", train_unsupervised.vec("Class"))

    train_features = model_nn.scoreDeepFeatures(train_unsupervised, 2)
    train_features._key = Key.make()
    train_features.add("Class", train_unsupervised.vec("Class"))

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

    ModelSerializationSupport.exportH2OModel(model_nn_dim, new File("C:/Users/admin-karim/Downloads/tmp/model/kutta.bin").toURI)

    val test_dim = model_nn.scoreDeepFeatures(test, 2)
    val test_dim_score = model_nn_dim.scoreAutoEncoder(test_dim, Key.make(), false)
    val result = confusionMatrix(test_dim_score, test, test_dim_score.anyVec.mean)
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
    asDataFrame(test).groupBy("Class", "predict").count.show    
    
    // Shutdown Spark cluster and H2O
    h2oContext.stop(stopSparkContext = true)
    spark.stop()
  }
}
