package com.packt.ScalaML

import org.apache.spark.ml.regression.{ RandomForestRegressor, RandomForestRegressionModel }
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.evaluation.RegressionMetrics

object RandomForestModelReuse {
  def main(args: Array[String]) {
    val spark = SparkSessionCreate.createSession()
    import spark.implicits._

    // Load the workflow back
    val cvModel = CrossValidatorModel.load("model/RF_model/")    

    // *****************************************
    println("Run prediction over test dataset")
    // *****************************************
    // Predicts and saves file ready for Kaggle!
    //if(!params.outputFile.isEmpty){
    cvModel.transform(Preproessing.testData)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "loss")
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("output/result_RF_reuse.csv")

    spark.stop()
  }
  
}