package com.packt.ScalaML.FraudDetection

import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import water.support.{H2OFrameSupport, ModelMetricsSupport, SparkContextSupport}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.year
import org.apache.spark.ml.feature.{ StringIndexer, VectorAssembler }
import org.apache.spark.ml.{ Pipeline, PipelineStage }
import scala.xml.persistent.SetStorage

import org.apache.spark.sql.functions._
import org.apache.spark.h2o._
import _root_.hex.FrameSplitter
import water.Key
import water.fvec.Frame
import _root_.hex.deeplearning.DeepLearning
import _root_.hex.deeplearning.DeepLearningModel
import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters
import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation
import java.io.File
import water.support.ModelSerializationSupport
import scala.reflect.api.materializeTypeTag

object BankingDataset extends SparkContextSupport with ModelMetricsSupport with H2OFrameSupport {
  /** Builds DeepLearning model. */
    def toCategorical(f: Frame, i: Int): Unit = {
    f.replace(i, f.vec(i).toCategoricalVec)
    f.update()
  }
  
  def buildDLModel(train: Frame, valid: Frame,
    epochs: Int = 1000, l1: Double = 0.001, l2: Double = 0.0,
    hidden: Array[Int] = Array[Int](10, 2, 10))(implicit h2oContext: H2OContext): DeepLearningModel = {
    import h2oContext.implicits._
    // Build a model
    val dlParams = new DeepLearningParameters()
    dlParams._train = train
    dlParams._valid = valid
    dlParams._response_column = 'label
    dlParams._epochs = epochs
    dlParams._l1 = l2
    dlParams._hidden = hidden

    // Create a job
    val dl = new DeepLearning(dlParams, water.Key.make("dlModel.hex"))
    dl.trainModel.get
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[3]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName(s"OneVsRestExample")
      .getOrCreate()

    spark.sqlContext.setConf("spark.sql.caseSensitive", "false");

    val trainDF = spark.read.
      option("inferSchema", "true")
      .format("com.databricks.spark.csv")
      .option("delimiter", ";")
      .option("header", "true")
      .load("data/bank-additional-full.csv")

    trainDF.show(10)
    val featureCols = Array("y", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "previous", "poutcome")

    //val featureCol = trainingDF.columns
    var indexers: Array[StringIndexer] = Array()

    featureCols.map { colName =>
      val index = new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(colName + "_indexed")
      indexers = indexers :+ index
    }

    val pipeline = new Pipeline().setStages(indexers)
    val indexedDF = pipeline.fit(trainDF).transform(trainDF)
    indexedDF.show()

    val indexer = new StringIndexer()
      .setInputCol("y_indexed")
      .setOutputCol("label")

    val labelIndexedDF = indexer.fit(indexedDF).transform(indexedDF)
    
    val newIndexedDF = labelIndexedDF.select("label", "job_indexed",
      "marital_indexed", "education_indexed", "default_indexed",
      "housing_indexed", "loan_indexed", "contact_indexed", "month_indexed",
      "day_of_week_indexed", "previous_indexed", "poutcome_indexed",
      "age", "duration", "campaign", "pdays", "previous", "emp_var_rate",
      "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed")

    val newFeatureCols = newIndexedDF.columns

    val assembler = new VectorAssembler()
      .setInputCols(newFeatureCols)
      .setOutputCol("features")

    val assembledDF = assembler.transform(newIndexedDF)
    assembledDF.show()
    
    val finalDF = assembledDF.select("label", "features")
    finalDF.show()

    implicit val h2oContext = H2OContext.getOrCreate(spark.sparkContext)
    import h2oContext.implicits._

    implicit val sqlContext = SparkSession.builder().getOrCreate().sqlContext
    import sqlContext.implicits._

    val H2ODF: H2OFrame = finalDF
    print(H2ODF)
    //val dataFrame = h2oContext.toH2OFrameKey(kuttaDF)

    // Split table
    val keys = Array[String]("train.hex", "valid.hex")
    val ratios = Array[Double](0.75)
    val frs = split(H2ODF, keys, ratios)
    //val (train, valid) = (frs(0), frs(1))
    //myDF.delete()
    
    val sf = new FrameSplitter(H2ODF, Array(0.6, 0.2), Array("train.hex", "valid.hex", "test.hex").map(Key.make[Frame](_)), null)
    water.H2O.submitTask(sf)
    val splits = sf.getResult
    val (train, valid, test) = (splits(0), splits(1), splits(2))

    // Build a model
    toCategorical(train, 0)
    toCategorical(valid, 0)
    toCategorical(test, 0)

    val dlModel = buildDLModel(train, valid)
    val auc = dlModel.auc()
    println("AUC: "+ dlModel.auc())
    println("Classfication Error: " + dlModel.classification_error())
    val result = dlModel.score(test)('predict)

    // Shutdown Spark cluster and H2O
    h2oContext.stop(stopSparkContext = true)
    spark.stop()
  }
}