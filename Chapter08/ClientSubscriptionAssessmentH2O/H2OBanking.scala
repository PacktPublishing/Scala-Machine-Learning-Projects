package com.packt.ScalaML.FraudDetection

import org.apache.spark.sql.{ DataFrame, SQLContext, SparkSession }
import water.support.{ H2OFrameSupport, ModelMetricsSupport, SparkContextSupport }
import org.apache.spark.sql._
import org.apache.spark.sql.functions.year
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
import vegas._
import vegas.sparkExt._
import _root_.hex.ModelMetricsBinomial

object BankingDataset2 extends SparkContextSupport with ModelMetricsSupport with H2OFrameSupport {
  /** Builds DeepLearning model. */
  def toCategorical(f: Frame, i: Int): Unit = {
    f.replace(i, f.vec(i).toCategoricalVec)
    f.update()
  }

  // Case classes for precision, recall, sensitivity and specificity
  case class r(precision: Double, recall: Double)
  case class r2(sensitivity: Double, specificity: Double)
  case class r3(tp: Double, fp: Double, tn: Double, fn: Double, th: Double)

  // Helper method for building DL model 
  def buildDLModel(train: Frame, valid: Frame,
    epochs: Int = 1000, l1: Double = 0.001, l2: Double = 0.0,
    hidden: Array[Int] = Array[Int](256, 256, 256))(implicit h2oContext: H2OContext): DeepLearningModel = {
    import h2oContext.implicits._

    // Build a model
    val dlParams = new DeepLearningParameters()
    dlParams._train = train
    dlParams._valid = valid
    dlParams._response_column = "y"
    dlParams._epochs = epochs
    dlParams._l1 = l1
    dlParams._hidden = hidden
    //dlParams._nfolds = 10

    // Create a job
    val dl = new DeepLearning(dlParams, water.Key.make("dlModel.hex"))
    dl.trainModel.get
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName(s"OneVsRestExample")
      .getOrCreate()

    spark.sqlContext.setConf("spark.sql.caseSensitive", "false");

    val trainDF = spark.read.
      option("inferSchema", "true").
      format("com.databricks.spark.csv").
      option("delimiter", ";").
      option("header", "true").
      load("data/bank-additional-full.csv")

    trainDF.show(10)
    val withoutDuration = trainDF.drop("duration")
    implicit val h2oContext = H2OContext.getOrCreate(spark.sparkContext)
    import h2oContext.implicits._
    implicit val sqlContext = spark.sqlContext
    import sqlContext.implicits._

    val H2ODF: H2OFrame = withoutDuration.orderBy(rand())
    //str to enum
    H2ODF.types.zipWithIndex.foreach(c => if (c._1.toInt == 2) toCategorical(H2ODF, c._2))
    val sf = new FrameSplitter(H2ODF, Array(0.6, 0.2), Array("train.hex", "valid.hex", "test.hex").map(Key.make[Frame](_)), null)
    water.H2O.submitTask(sf)
    val splits = sf.getResult
    val (train, valid, test) = (splits(0), splits(1), splits(2))
    val dlModel = buildDLModel(train, valid)

    val trainAUC = dlModel.auc()
    println("AUC: " + trainAUC)
    println("Trainin classfication error" + dlModel.classification_error())

    // Model evaluation on test set
    val result = dlModel.score(test)('predict)
    val trainMetrics = ModelMetricsSupport.modelMetrics[ModelMetricsBinomial](dlModel, test)
    println(s"Training AUC: ${trainMetrics.auc}")
    println(trainMetrics)

    //Actual vs predicted values
    result.add("actual", test.vec("y"))
    val predict_actualDF = h2oContext.asDataFrame(result)
    predict_actualDF.groupBy("actual", "predict").count.show

    Vegas()
      .withDataFrame(predict_actualDF)
      .mark(Bar)
      .encodeY(field = "*", dataType = Quantitative, AggOps.Count, axis = Axis(title = "", format = ".2f"), hideAxis = true)
      .encodeX("actual", Ord).encodeColor("predict", Nominal, scale = Scale(rangeNominals = List("#FF2800", "#1C39BB")))
      .configMark(stacked = StackOffset.Normalize)
      .show

    // Aria under the curve
    val auc = trainMetrics._auc
    //tp,fp,tn,fn
    val metrics = auc._tps.zip(auc._fps).zipWithIndex.map(x => x match { case ((a, b), c) => (a, b, c) })
    val fullmetrics = metrics.map(_ match { case (a, b, c) => (a, b, auc.tn(c), auc.fn(c)) })
    val precisions = fullmetrics.map(_ match { case (tp, fp, tn, fn) => tp / (tp + fp) })
    val recalls = fullmetrics.map(_ match { case (tp, fp, tn, fn) => tp / (tp + fn) })

    val rows = for (i <- 0 until recalls.length) yield r(precisions(i), recalls(i))
    //val rowsRDD = spark.sparkContext.p
    val precision_recall = rows.toDF()

    //precision vs recall
    Vegas("ROC", width = 800, height = 600).withDataFrame(precision_recall).mark(Line).encodeX("re-call", Quantitative).encodeY("precision", Quantitative).show

    //sensitivity_specificity
    val sensitivity = fullmetrics.map(_ match { case (tp, fp, tn, fn) => tp / (tp + fn) })
    val specificity = fullmetrics.map(_ match { case (tp, fp, tn, fn) => tn / (tn + fp) })

    val rows2 = for (i <- 0 until specificity.length) yield r2(sensitivity(i), specificity(i))
    val sensitivity_specificity = rows2.toDF
    Vegas("sensitivity_specificity", width = 800, height = 600).withDataFrame(sensitivity_specificity).mark(Line).encodeX("specificity", Quantitative).encodeY("sensitivity", Quantitative).show

    //Threshold vs tp,tn,fp,fn
    val withTh = auc._tps.zip(auc._fps).zipWithIndex.map(x => x match { case ((a, b), c) => (a, b, auc.tn(c), auc.fn(c), auc._ths(c)) })

    val rows3 = for (i <- 0 until withTh.length) yield r3(withTh(i)._1, withTh(i)._2, withTh(i)._3, withTh(i)._4, withTh(i)._5)
    Vegas("tp", width = 800, height = 600).withDataFrame(rows3.toDF).mark(Line).encodeX("th", Quantitative).encodeY("tp", Quantitative).show

    Vegas("fp", width = 800, height = 600).withDataFrame(rows3.toDF).mark(Line).encodeX("th", Quantitative).encodeY("fp", Quantitative).show
    Vegas("tn", width = 800, height = 600).withDataFrame(rows3.toDF).mark(Line).encodeX("th", Quantitative).encodeY("tn", Quantitative).show
    Vegas("fn", width = 800, height = 600).withDataFrame(rows3.toDF).mark(Line).encodeX("th", Quantitative).encodeY("fn", Quantitative).show

    // Shutdown Spark cluster and H2O
    h2oContext.stop(stopSparkContext = true)
    spark.stop()
  }
}
