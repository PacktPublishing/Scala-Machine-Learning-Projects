package com.packt.ScalaML.MovieRecommendation

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SQLImplicits
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import scala.Tuple2
import org.apache.spark.rdd.RDD

object MovieRecommendation {  
  //Compute the RMSE to evaluate the model. Less the RMSE better the model and it's prediction capability. 
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], implicitPrefs: Boolean): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map { x => ((x.user, x.product), x.rating)
    }.join(data.map(x => ((x.user, x.product), x.rating))).values
    if (implicitPrefs) {
      println("(Prediction, Rating)")
      println(predictionsAndRatings.take(5).mkString("\n"))
    }
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("JavaLDAExample")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/").
      getOrCreate()

    val ratigsFile = "data/ratings.csv"
    val df1 = spark.read.format("com.databricks.spark.csv").option("header", true).load(ratigsFile)

    val ratingsDF = df1.select(df1.col("userId"), df1.col("movieId"), df1.col("rating"), df1.col("timestamp"))
    ratingsDF.show(false)

    val moviesFile = "data/movies.csv"
    val df2 = spark.read.format("com.databricks.spark.csv").option("header", "true").load(moviesFile)

    val moviesDF = df2.select(df2.col("movieId"), df2.col("title"), df2.col("genres"))
    moviesDF.show(false)

    ratingsDF.createOrReplaceTempView("ratings")
    moviesDF.createOrReplaceTempView("movies")

    /*
		 * Explore and Query with Spark DataFrames		 */

    val numRatings = ratingsDF.count()
    val numUsers = ratingsDF.select(ratingsDF.col("userId")).distinct().count()
    val numMovies = ratingsDF.select(ratingsDF.col("movieId")).distinct().count()
    println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.")

    // Get the max, min ratings along with the count of users who have rated a movie.
    val results = spark.sql("select movies.title, movierates.maxr, movierates.minr, movierates.cntu "
      + "from(SELECT ratings.movieId,max(ratings.rating) as maxr,"
      + "min(ratings.rating) as minr,count(distinct userId) as cntu "
      + "FROM ratings group by ratings.movieId) movierates "
      + "join movies on movierates.movieId=movies.movieId " + "order by movierates.cntu desc");

    results.show(false)

    // Show the top 10 most-active users and how many times they rated a movie
    val mostActiveUsersSchemaRDD = spark.sql("SELECT ratings.userId, count(*) as ct from ratings "
                                              + "group by ratings.userId order by ct desc limit 10")
    mostActiveUsersSchemaRDD.show(false)

    // Find the movies that user 668 rated higher than 4
    val results2 = spark.sql(
                "SELECT ratings.userId, ratings.movieId," 
                + "ratings.rating, movies.title FROM ratings JOIN movies "
                + "ON movies.movieId=ratings.movieId " 
                + "where ratings.userId=668 and ratings.rating > 4")
        
    results2.show(false)

    // Randomly split ratings RDD into training data RDD (75%) and test data RDD (25%)
    val splits = ratingsDF.randomSplit(Array(0.75, 0.25), seed = 12345L)
    val (trainingData, testData) = (splits(0), splits(1))
    
    val numTraining = trainingData.count()
    val numTest = testData.count()
    println("Training: " + numTraining + " test: " + numTest)

    val ratingsRDD = trainingData.rdd.map(row => {
      val userId = row.getString(0)
      val movieId = row.getString(1)
      val ratings = row.getString(2)
      Rating(userId.toInt, movieId.toInt, ratings.toDouble)
    })

    val testRDD = testData.rdd.map(row => {
      val userId = row.getString(0)
      val movieId = row.getString(1)
      val ratings = row.getString(2)
      Rating(userId.toInt, movieId.toInt, ratings.toDouble)
    })

    /*
		 * Using ALS with the Movie Ratings Data: build a ALS user product matrix model with rank=20, iterations=10 
		 * The training is done via the Collaborative Filtering algorithm Alternating Least Squares (ALS). 
		 * Essentially this technique predicts missing ratings for specific users for specific
		 *  movies based on ratings for those movies from other users who did similar ratings for other movies. Read the documentation for more details.
		 * */
    
    val rank = 20
    val numIterations = 10
    val lambda = 0.01
    val alpha = 10
    
    val model = new ALS()
                  .setIterations(15)
                  .setBlocks(-1)
                  .setAlpha(1.0)
                  .setLambda(0.10)
                  .setRank(10)
                  .setSeed(12345L)
                  .setImplicitPrefs(false)
                  .run(ratingsRDD)
                  
    //Saving the model for future use
    val savedALSModel = model.save(spark.sparkContext, "model/MovieRecomModel")

    // Making Predictions. Get the top 6 movie predictions for user 668
    println("Rating:(UserID, MovieID, Rating)")
    println("----------------------------------")
    val topRecsForUser = model.recommendProducts(668, 6)
    for (rating <- topRecsForUser) {
      println(rating.toString())
    }
    println("----------------------------------")

    /*
		 * Evaluating the Model: 
		 * In order to verify the quality of the models Root Mean Squared Error (RMSE) is used to measure the 
		 * differences between values predicted by a model and the values actually observed. 
		 * The smaller the calculated error, the better the model. In order to test the quality of the model, the test data is used (which was split above).
		 * RMSD is a good measure of accuracy, but only to compare forecasting errors of different models for a particular variable and not between variables, as it is scale-dependent
		 */

    var rmseTest = computeRmse(model, testRDD, true)
    println("Test RMSE: = " + rmseTest) //Less is better

    //Movie recommendation for a specific user. Get the top 6 movie predictions for user 668
    println("Recommendations: (MovieId => Rating)")
    println("----------------------------------")
    val recommendationsUser = model.recommendProducts(668, 6)
    recommendationsUser.map(rating => (rating.product, rating.rating)).foreach(println)
    println("----------------------------------")

    spark.stop()
  }
}
