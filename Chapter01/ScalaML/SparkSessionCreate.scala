package com.packt.ScalaML

import org.apache.spark.sql.SparkSession

object SparkSessionCreate {
  def createSession(): SparkSession = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
      .appName(s"OneVsRestExample")
      .getOrCreate()

    return spark
  }
}