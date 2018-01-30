package com.packt.ScalaML.ChrunPrediction

import org.apache.spark._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset

/*
 * Dataset schema
State	
Account length	
Area code	
International plan	
Voice mail plan	
Number vmail messages	
Total day minutes	
Total day calls	
Total day charge	
Total eve minutes	
Total eve calls	Total eve charge	
Total night minutes	
Total night calls	
Total night charge	
Total intl minutes	
Total intl calls	
Total intl charge	
Customer service calls	
Churn
 */

object Preprocessing {
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

  val spark: SparkSession = SparkSessionCreate.createSession("ChurnPredictionRandomForest")
  import spark.implicits._

  val trainSet: Dataset[CustomerAccount] = spark.read.
    option("inferSchema", "false")
    .format("com.databricks.spark.csv")
    .schema(schema)
    .load("data/churn-bigml-80.csv")
    .as[CustomerAccount]

  val statsDF = trainSet.describe()
  statsDF.show()
  trainSet.cache()

  trainSet.groupBy("churn").sum("total_international_num_calls").show()
  trainSet.groupBy("churn").sum("total_international_charge").show()

  val testSet: Dataset[CustomerAccount] = spark.read.
    option("inferSchema", "false")
    .format("com.databricks.spark.csv")
    .schema(schema)
    .load("data/churn-bigml-20.csv")
    .as[CustomerAccount]

  testSet.describe()
  testSet.cache()

  trainSet.printSchema()
  trainSet.show()

  trainSet.createOrReplaceTempView("UserAccount")
  spark.catalog.cacheTable("UserAccount")

  /////////////// Feature engineering
  spark.sqlContext.sql("SELECT churn, SUM(total_day_mins) + SUM(total_evening_mins) + SUM(total_night_mins) + SUM(total_international_mins) as Total_minutes FROM UserAccount GROUP BY churn").show()
  spark.sqlContext.sql("SELECT churn, SUM(total_day_charge) as TDC, SUM(total_evening_charge) as TEC, SUM(total_night_charge) as TNC, SUM(total_international_charge) as TIC, SUM(total_day_charge) + SUM(total_evening_charge) + SUM(total_night_charge) + SUM(total_international_charge) as Total_charge FROM UserAccount GROUP BY churn ORDER BY Total_charge DESC").show()
  trainSet.groupBy("churn").count.show()
  spark.sqlContext.sql("SELECT churn,SUM(total_international_num_calls) as Total_intl_call FROM UserAccount GROUP BY churn").show()

  val fractions = Map("False" -> 0.1675, "True" -> 1.0)

  //Here we're keeping all instances of the Churn=True class, but downsampling the Churn=False class to a fraction of 388/2278.
  val churnDF = trainSet.stat.sampleBy("churn", fractions, 123456L)

  churnDF.groupBy("churn").count.show()

  val trainDF = churnDF
    .drop("state_code")
    .drop("area_code")
    .drop("voice_mail_plan")
    .drop("total_day_charge")
    .drop("total_evening_charge")

  println(trainDF.count)
  trainDF.select("account_length", "international_plan", "num_voice_mail", "total_day_calls", "total_international_num_calls", "churn").show(10)
}