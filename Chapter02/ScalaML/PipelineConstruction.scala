package com.packt.ScalaML.ChrunPrediction

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}

object PipelineConstruction {
    // Index labels, adding metadata to the label column. Fit on whole dataset to include all labels in index.
    val ipindexer = new StringIndexer()
      .setInputCol("international_plan")
      .setOutputCol("iplanIndex")

    val labelindexer = new StringIndexer()
      .setInputCol("churn")
      .setOutputCol("label")
      
    val featureCols = Array("account_length", "iplanIndex", "num_voice_mail", "total_day_mins", "total_day_calls", "total_evening_mins", "total_evening_calls", "total_night_mins", "total_night_calls", "total_international_mins", "total_international_calls", "total_international_num_calls")

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features") 
}