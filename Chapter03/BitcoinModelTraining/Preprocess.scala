package com.packt.ScalaML.BitCoin

import java.io.{ BufferedWriter, File, FileWriter }
import org.apache.spark.sql.types.{ DoubleType, IntegerType, StructField, StructType }
import org.apache.spark.sql.{ DataFrame, Row, SparkSession }
import scala.collection.mutable.ListBuffer

object Preprocess {
  //how many of first rows are omitted
    val dropFirstCount: Int = 612000

    def rollingWindow(data: DataFrame, window: Int, xFilename: String, yFilename: String): Unit = {
      var i = 0
      val xWriter = new BufferedWriter(new FileWriter(new File(xFilename)))
      val yWriter = new BufferedWriter(new FileWriter(new File(yFilename)))

      val zippedData = data.rdd.zipWithIndex().collect()
      System.gc()
      val dataStratified = zippedData.drop(dropFirstCount) //todo slice fisrt 614K
      while (i < (dataStratified.length - window)) {
        val x = dataStratified
          .slice(i, i + window)
          .map(r => r._1.getAs[Double]("Delta")).toList
        val y = dataStratified.apply(i + window)._1.getAs[Integer]("label")
        val stringToWrite = x.mkString(",")
        xWriter.write(stringToWrite + "\n")
        yWriter.write(y + "\n")

        i += 1
        if (i % 10 == 0) {
          xWriter.flush()
          yWriter.flush()
        }
      }

      xWriter.close()
      yWriter.close()
    }
    
  def main(args: Array[String]): Unit = {
    //todo modify these variables to match desirable files
    val priceDataFileName: String = "C:/Users/admin-karim/Desktop/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv"
    val outputDataFilePath: String = "output/scala_test_x.csv"
    val outputLabelFilePath: String = "output/scala_test_y.csv"

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName("Bitcoin Preprocessing")
      .getOrCreate()

    val data = spark.read.format("com.databricks.spark.csv").option("header", "true").load(priceDataFileName)
    data.show(10)
    println((data.count(), data.columns.size))

    val dataWithDelta = data.withColumn("Delta", data("Close") - data("Open"))

    import org.apache.spark.sql.functions._
    import spark.sqlContext.implicits._

    val dataWithLabels = dataWithDelta.withColumn("label", when($"Close" - $"Open" > 0, 1).otherwise(0))
    rollingWindow(dataWithLabels, 22, outputDataFilePath, outputLabelFilePath)    
    spark.stop()
  }
}