package com.packt.ScalaML.TopicModelling

import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.clustering.{ DistributedLDAModel, LDA }

object LDAModelReuse {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "data/")
      .appName(s"OneVsRestExample")
      .getOrCreate()

    //Restoring the model for reuse
    val savedLDAModel = DistributedLDAModel.load(spark.sparkContext, "model/LDATrainedModel/")

    val lda = new LDAforTM() // actual computations are done here
    val defaultParams = Params().copy(input = "data/4UK1UkTX.csv", savedLDAModel) // Loading the parameters to train the LDA model
    lda.run(defaultParams) // Training the LDA model with the default parameters. 
    spark.stop()
  }
}
