package com.packt.ScalaML.ChrunPrediction

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{ BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel }
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.max
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

import org.apache.spark._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset

import org.apache.spark.ml.linalg.{ Matrix, Vectors }
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row

object Describe {
  case class CustomerAccount(state_code: String, account_length: Integer, area_code: String,
    international_plan: String, voice_mail_plan: String, num_voice_mail: Double,
    total_day_mins: Double, total_day_calls: Double, total_day_charge: Double,
    total_evening_mins: Double, total_evening_calls: Double, total_evening_charge: Double,
    total_night_mins: Double, total_night_calls: Double, total_night_charge: Double,
    total_international_mins: Double, total_international_calls: Double, total_international_charge: Double,
    total_international_num_calls: Double, churn: String)

  val schema = StructType(Array(
    StructField("state_code", StringType, true),
    StructField("account_length", IntegerType, true),
    StructField("area_code", StringType, true),
    StructField("international_plan", StringType, true),
    StructField("voice_mail_plan", StringType, true),
    StructField("num_voice_mail", DoubleType, true),
    StructField("total_day_mins", DoubleType, true),
    StructField("total_day_calls", DoubleType, true),
    StructField("total_day_charge", DoubleType, true),
    StructField("total_evening_mins", DoubleType, true),
    StructField("total_evening_calls", DoubleType, true),
    StructField("total_evening_charge", DoubleType, true),
    StructField("total_night_mins", DoubleType, true),
    StructField("total_night_calls", DoubleType, true),
    StructField("total_night_charge", DoubleType, true),
    StructField("total_international_mins", DoubleType, true),
    StructField("total_international_calls", DoubleType, true),
    StructField("total_international_charge", DoubleType, true),
    StructField("total_international_num_calls", DoubleType, true),
    StructField("churn", StringType, true)))

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName("Desribe")
      .getOrCreate()

    spark.conf.set("spark.debug.maxToStringFields", 10000)
    val DEFAULT_MAX_TO_STRING_FIELDS = 2500
    if (SparkEnv.get != null) {
      SparkEnv.get.conf.getInt("spark.debug.maxToStringFields", DEFAULT_MAX_TO_STRING_FIELDS)
    } else {
      DEFAULT_MAX_TO_STRING_FIELDS
    }
    import spark.implicits._

    val trainSet: Dataset[CustomerAccount] = spark.read.
      option("inferSchema", "false")
      .format("com.databricks.spark.csv")
      .schema(schema)
      .load("data/churn-bigml-80.csv")
      .as[CustomerAccount]

    val statsDF = trainSet.describe()   
    statsDF.show()

    trainSet.createOrReplaceTempView("UserAccount")
    spark.catalog.cacheTable("UserAccount")
    
    spark.sqlContext.sql("SELECT churn, SUM(total_day_mins) + SUM(total_evening_mins) + SUM(total_night_mins) + SUM(total_international_mins) as Total_minutes FROM UserAccount GROUP BY churn").show()
    spark.sqlContext.sql("SELECT churn, SUM(total_day_charge) as TDC, SUM(total_evening_charge) as TEC, SUM(total_night_charge) as TNC, SUM(total_international_charge) as TIC, SUM(total_day_charge) + SUM(total_evening_charge) + SUM(total_night_charge) + SUM(total_international_charge) as Total_charge FROM UserAccount GROUP BY churn ORDER BY Total_charge DESC").show()
    trainSet.groupBy("churn").count.show()
    spark.sqlContext.sql("SELECT churn,SUM(total_international_num_calls) FROM UserAccount GROUP BY churn")
    
  }
}