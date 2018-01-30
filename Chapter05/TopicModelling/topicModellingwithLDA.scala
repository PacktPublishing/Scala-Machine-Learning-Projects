package com.packt.ScalaML.TopicModelling

import edu.stanford.nlp.process.Morphology
import edu.stanford.nlp.simple.Document
import org.apache.log4j.{Level, Logger}
import scala.collection.JavaConversions._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, EMLDAOptimizer, LDA, OnlineLDAOptimizer, LDAModel}
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}

object topicModellingwithLDA {
  def main(args: Array[String]): Unit = {
    val lda = new LDAforTM() // actual computations are done here
    val defaultParams = Params().copy(input = "data/docs/") // Loading the parameters to train the LDA model
    lda.run(defaultParams) // Training the LDA model with the default parameters. 
  }
}

//Setting the parameters before training the LDA model
case class Params(var input: String = "", var ldaModel: LDAModel = null,
  k: Int = 5,
  maxIterations: Int = 100,
  docConcentration: Double = 5,
  topicConcentration: Double = 5,
  vocabSize: Int = 2900000,
  stopwordFile: String = "data/docs/stopWords.txt",
  algorithm: String = "em",
  checkpointDir: Option[String] = None,
  checkpointInterval: Int = 100)

// actual computations for topic modeling are done here
class LDAforTM() {
  val spark = SparkSession
    .builder
    .master("local[*]")
    .config("spark.sql.warehouse.dir", "data/")
    .appName(s"OneVsRestExample")
    .getOrCreate()

  def run(params: Params): Unit = {
    Logger.getRootLogger.setLevel(Level.WARN)

    // Load documents, and prepare them for LDA.
    val preprocessStart = System.nanoTime()
    val (corpus, vocabArray, actualNumTokens) = preprocess(params.input, params.vocabSize, params.stopwordFile)

    val actualCorpusSize = corpus.count()
    val actualVocabSize = vocabArray.length
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9
    corpus.cache() //will be reused later steps

    println()
    println("Training corpus summary:")
    println("-------------------------------")
    println("Training set size: " + actualCorpusSize + " documents")
    println("Vocabulary size: " + actualVocabSize + " terms")
    println("Number of tockens: " + actualNumTokens + " tokens")
    println("Preprocessing time: " + preprocessElapsed + " sec")
    println("-------------------------------")
    println()

    // Instantiate an LDA model
    val lda = new LDA()

    val optimizer = params.algorithm.toLowerCase match {
      case "em" => new EMLDAOptimizer
      // add (1.0 / actualCorpusSize) to MiniBatchFraction be more robust on tiny datasets.
      case "online" => new OnlineLDAOptimizer().setMiniBatchFraction(0.05 + 1.0 / actualCorpusSize)
      case _ => throw new IllegalArgumentException("Only em, online are supported but got ${params.algorithm}.")
    }

    lda.setOptimizer(optimizer)
      .setK(params.k)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .setCheckpointInterval(params.checkpointInterval)

    if (params.checkpointDir.nonEmpty) {
      spark.sparkContext.setCheckpointDir(params.checkpointDir.get)
    }

    val startTime = System.nanoTime()

    //Start training the LDA model using the training corpus 
    params.ldaModel = lda.run(corpus)
    val elapsed = (System.nanoTime() - startTime) / 1e9
        
    //Saving the model for future use
    params.ldaModel.save(spark.sparkContext, "model/LDATrainedModel")

    println("Finished training LDA model.  Summary:")
    println("Training time: " + elapsed + " sec")

    if (params.ldaModel.isInstanceOf[DistributedLDAModel]) {
      val distLDAModel = params.ldaModel.asInstanceOf[DistributedLDAModel]
      val avgLogLikelihood = distLDAModel.logLikelihood / actualCorpusSize.toDouble
      println("The average log likelihood of the training data: " + avgLogLikelihood)
      println()
    }

    // Print the topics, showing the top-weighted terms for each topic.
    val topicIndices = params.ldaModel.describeTopics(maxTermsPerTopic = 10)
    println(topicIndices.length)
    val topics = topicIndices.map { case (terms, termWeights) => terms.zip(termWeights).map { case (term, weight) => (vocabArray(term.toInt), weight) } }

    var sum = 0.0
    println(s"${params.k} topics:")
    topics.zipWithIndex.foreach {
      case (topic, i) =>
        println(s"TOPIC $i")
        println("------------------------------")
        topic.foreach {
          case (term, weight) =>
            term.replaceAll("\\s", "")
            println(s"$term\t$weight")
            sum = sum + weight
        }
        println("----------------------------")
        println("weight: " + sum)
        println()
    }
    spark.stop()
  }

  //Pre-processing of the raw texts
  import org.apache.spark.sql.functions._
  def preprocess(paths: String, vocabSize: Int, stopwordFile: String): (RDD[(Long, Vector)], Array[String], Long) = {
    import spark.implicits._
    //Reading the Whole Text Files
    val initialrdd = spark.sparkContext.wholeTextFiles(paths).map(_._2)
    initialrdd.cache()

    val rdd = initialrdd.mapPartitions { partition =>
      val morphology = new Morphology()
      partition.map { value => helperForLDA.getLemmaText(value, morphology) }
    }.map(helperForLDA.filterSpecialCharacters)

    rdd.cache()
    initialrdd.unpersist()
    val df = rdd.toDF("docs")
    df.show()

    //Customizing the stop words
    val customizedStopWords: Array[String] = if (stopwordFile.isEmpty) {
      Array.empty[String]
    } else {
      val stopWordText = spark.sparkContext.textFile(stopwordFile).collect()
      stopWordText.flatMap(_.stripMargin.split(","))
    }

    //Tokenizing using the RegexTokenizer
    val tokenizer = new RegexTokenizer().setInputCol("docs").setOutputCol("rawTokens")

    //Removing the Stop-words using the Stop Words remover
    val stopWordsRemover = new StopWordsRemover().setInputCol("rawTokens").setOutputCol("tokens")
    stopWordsRemover.setStopWords(stopWordsRemover.getStopWords ++ customizedStopWords)

    //Converting the Tokens into the CountVector
    val countVectorizer = new CountVectorizer().setVocabSize(vocabSize).setInputCol("tokens").setOutputCol("features")

    //Setting up the pipeline
    val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, countVectorizer))

    val model = pipeline.fit(df)
    
    val documents = model.transform(df).select("features").rdd.map {
      case Row(features: MLVector) => Vectors.fromML(features)
    }.zipWithIndex().map(_.swap)

    //Returning the vocabulary and  tocken count pairs
    (documents, model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary, documents.map(_._2.numActives).sum().toLong)
  }
}

object helperForLDA {
  def filterSpecialCharacters(document: String) = document.replaceAll("""[! @ # $ % ^ & * ( ) _ + - − , " ' ; : . ` ? --]""", " ")

  def getLemmaText(document: String, morphology: Morphology) = {
    val string = new StringBuilder()
    val value = new Document(document).sentences().toList.flatMap { a =>
      val words = a.words().toList
      val tags = a.posTags().toList
      (words zip tags).toMap.map { a =>
        val newWord = morphology.lemma(a._1, a._2)
        val addedWoed = if (newWord.length > 3) {
          newWord
        } else { "" }
        string.append(addedWoed + " ")
      }
    }
    string.toString()
  }
}