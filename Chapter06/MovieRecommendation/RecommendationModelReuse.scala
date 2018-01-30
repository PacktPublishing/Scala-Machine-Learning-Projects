package com.packt.ScalaML.MovieRecommendation

import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import scala.Tuple2
import org.apache.spark.rdd.RDD

object RecommendationModelReuse {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("JavaLDAExample")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/").
      getOrCreate()

    val ratigsFile = "data/ratings.csv"
    val ratingDF = spark.read.format("com.databricks.spark.csv").option("header", true).load(ratigsFile)
    val selectedRatingsDF = ratingDF.select(ratingDF.col("userId"), ratingDF.col("movieId"), ratingDF.col("rating"), ratingDF.col("timestamp"))

    // Randomly split ratings RDD into training data RDD (75%) and test data RDD (25%)
    val splits = selectedRatingsDF.randomSplit(Array(0.75, 0.25), seed = 12345L)
    val testData = splits(1)

    val testRDD = testData.rdd.map(row => {
      val userId = row.getString(0)
      val movieId = row.getString(1)
      val ratings = row.getString(2)
      Rating(userId.toInt, movieId.toInt, ratings.toDouble)
    })

    //Load the workflow back
    val same_model = MatrixFactorizationModel.load(spark.sparkContext, "model/MovieRecomModel/")

    // Making Predictions. Get the top 6 movie predictions for user 668
    println("Rating:(UserID, MovieID, Rating)")
    println("----------------------------------")
    val topRecsForUser = same_model.recommendProducts(458, 10)
    for (rating <- topRecsForUser) {
      println(rating.toString())
    }
    println("----------------------------------")

    val rmseTest = MovieRecommendation.computeRmse(same_model, testRDD, true)
    println("Test RMSE: = " + rmseTest) //Less is better

    //Movie recommendation for a specific user. Get the top 6 movie predictions for user 668
    println("Recommendations: (MovieId => Rating)")
    println("----------------------------------")
    val recommendationsUser = same_model.recommendProducts(458, 10)
    recommendationsUser.map(rating => (rating.product, rating.rating)).foreach(println)
    println("----------------------------------")

    spark.stop()
  }
}
