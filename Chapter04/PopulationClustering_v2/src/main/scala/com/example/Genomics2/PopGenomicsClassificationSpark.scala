package com.example.Genomics2

import hex.FrameSplitter
import org.apache.spark.SparkContext
import org.apache.spark.h2o.H2OContext
import org.bdgenomics.adam.rdd.ADAMContext._
import org.bdgenomics.formats.avro.{ Genotype, GenotypeAllele }
import water.{ Job, Key }
import water.fvec.Frame

import org.apache.spark.h2o._
import java.io.File
import java.io._
import scala.collection.JavaConverters._
import scala.collection.immutable.Range.inclusive
import scala.io.Source

import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.{ IntegerType, StringType, StructField, StructType }
import org.apache.spark.ml.feature.{ VectorAssembler, Normalizer }
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.{ Pipeline }
import org.apache.spark.ml.classification.{ RandomForestClassifier, RandomForestClassificationModel }
import org.apache.spark.ml.evaluation.{ MulticlassClassificationEvaluator }
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }

object PopGenomicsClassificationSpark {
  def main(args: Array[String]): Unit = {
    val genotypeFile = "C:/Users/admin-karim/Downloads/genotypes.vcf"
    val panelFile = "C:/Users/admin-karim/Downloads/genotypes.panel"

    val spark:SparkSession =  SparkSession
                              .builder()
                               .appName("PopStrat")
                                .master("local[*]")
                                 .config("spark.sql.warehouse.dir", "C:/Exp/")
                                  .getOrCreate()
                                            
    val sc: SparkContext = spark.sparkContext

    // Create a set of the populations that we want to predict
    // Then create a map of sample ID -> population so that we can filter out the samples we're not interested in
    //val populations = Set("GBR", "ASW", "FIN", "CHB", "CLM")
    val populations = Set("FIN", "GBR", "ASW", "CHB", "CLM")

    def extract(file: String,
      filter: (String, String) => Boolean): Map[String, String] = {
      Source
        .fromFile(file)
        .getLines()
        .map(line => {
          val tokens = line.split(Array('\t', ' ')).toList
          tokens(0) -> tokens(1)
        })
        .toMap
        .filter(tuple => filter(tuple._1, tuple._2))
    }

    val panel: Map[String, String] = extract(
      panelFile,
      (sampleID: String, pop: String) => populations.contains(pop))

    // Load the ADAM genotypes from the parquet file(s)
    // Next, filter the genotypes so that we're left with only those in the populations we're interested in
    val allGenotypes: RDD[Genotype] = sc.loadGenotypes(genotypeFile).rdd
    //allGenotypes.adamParquetSave("output")
    val genotypes: RDD[Genotype] = allGenotypes.filter(genotype => {
      panel.contains(genotype.getSampleId)
    })

    // Convert the Genotype objects to our own SampleVariant objects to try and conserve memory
    case class SampleVariant(sampleId: String,
      variantId: Int,
      alternateCount: Int)
      
    def variantId(genotype: Genotype): String = {
      val name = genotype.getVariant.getContigName
      val start = genotype.getVariant.getStart
      val end = genotype.getVariant.getEnd
      s"$name:$start:$end"
    }

    def alternateCount(genotype: Genotype): Int = {
      genotype.getAlleles.asScala.count(_ != GenotypeAllele.REF)
    }

    def toVariant(genotype: Genotype): SampleVariant = {
      // Intern sample IDs as they will be repeated a lot
      new SampleVariant(genotype.getSampleId.intern(),
        variantId(genotype).hashCode(),
        alternateCount(genotype))
    }

    val variantsRDD: RDD[SampleVariant] = genotypes.map(toVariant)
    //println(s"Variant RDD: " + variantsRDD.first())

    // Group the variants by sample ID so we can process the variants sample-by-sample
    // Then get the total number of samples. This will be used to find variants that are missing for some samples.
    // Group the variants by variant ID and filter out those variants that are missing from some samples
    val variantsBySampleId: RDD[(String, Iterable[SampleVariant])] =
      variantsRDD.groupBy(_.sampleId)
    val sampleCount: Long = variantsBySampleId.count()
    println("Found " + sampleCount + " samples")

    val variantsByVariantId: RDD[(Int, Iterable[SampleVariant])] =
      variantsRDD.groupBy(_.variantId).filter {
        case (_, sampleVariants) => sampleVariants.size == sampleCount
      }

    // Make a map of variant ID -> count of samples with an alternate count of greater than zero
    // then filter out those variants that are not in our desired frequency range. The objective here is simply to
    // reduce the number of dimensions in the data set to make it easier to train the model.
    // The specified range is fairly arbitrary and was chosen based on the fact that it includes a reasonable
    // number of variants, but not too many.
    val variantFrequencies: collection.Map[Int, Int] = variantsByVariantId
      .map {
        case (variantId, sampleVariants) =>
          (variantId, sampleVariants.count(_.alternateCount > 0))
      }
      .collectAsMap()

    val permittedRange = inclusive(11, 11)
    val filteredVariantsBySampleId: RDD[(String, Iterable[SampleVariant])] =
      variantsBySampleId.map {
        case (sampleId, sampleVariants) =>
          val filteredSampleVariants = sampleVariants.filter(
            variant =>
              permittedRange.contains(
                variantFrequencies.getOrElse(variant.variantId, -1)))
          (sampleId, filteredSampleVariants)
      }

    //println(s"Filtered Variant RDD: " + filteredVariantsBySampleId.first())

    // Sort the variants for each sample ID. Each sample should now have the same number of sorted variants.
    // All items in the RDD should now have the same variants in the same order so we can just use the first
    // one to construct our header
    // Next construct the rows of our SchemaRDD from the variants
    val sortedVariantsBySampleId: RDD[(String, Array[SampleVariant])] =
      filteredVariantsBySampleId.map {
        case (sampleId, variants) =>
          (sampleId, variants.toArray.sortBy(_.variantId))
      }

    println(s"Sorted by Sample ID RDD: " + sortedVariantsBySampleId.first())

    val header = StructType(
      Seq(StructField("Region", StringType)) ++
        sortedVariantsBySampleId
        .first()
        ._2
        .map(variant => {
          StructField(variant.variantId.toString, IntegerType)
        }))

    val rowRDD: RDD[Row] = sortedVariantsBySampleId.map {
      case (sampleId, sortedVariants) =>
        val region: Array[String] = Array(panel.getOrElse(sampleId, "Unknown"))
        val alternateCounts: Array[Int] = sortedVariants.map(_.alternateCount)
        Row.fromSeq(region ++ alternateCounts)
    }

    // Create the SchemaRDD from the header and rows and convert the SchemaRDD into a H2O dataframe
    val sqlContext = spark.sqlContext
    val schemaDF = sqlContext.createDataFrame(rowRDD, header)
    schemaDF.printSchema()
    schemaDF.show(10)

    val featureCols = schemaDF.columns.drop(1)

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val assembleDF = assembler.transform(schemaDF).select("features", "Region")
    assembleDF.show()

    /*
    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(50)
      .fit(assembleDF)

    val pcaDF = pca.transform(assembleDF).select("pcaFeatures", "Region").withColumnRenamed("pcaFeatures", "features")//.withColumnRenamed("Region", "label")
    pcaDF.show()
    * 
    */
    
    
    val indexer = new StringIndexer()
      .setInputCol("Region")
      .setOutputCol("label")

    val indexedDF = indexer.fit(assembleDF).transform(assembleDF).select("features", "label")
    println("Indeexed: ")
    indexedDF.show(10)

    val seed = 12345L
    val splits = indexedDF.randomSplit(Array(0.75, 0.25), seed)
    val (trainDF, testDF) = (splits(0), splits(1))

    trainDF.cache
    testDF.cache

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setSeed(1234567L)

    // Search through decision tree's maxDepth parameter for best model
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxDepth, 3 :: 5 :: 15 :: 20 :: 25 :: 30 :: Nil)
      .addGrid(rf.featureSubsetStrategy, "auto" :: "all" :: Nil)
      .addGrid(rf.impurity, "gini" :: "entropy" :: Nil)
      .addGrid(rf.maxBins, 3 :: 5 :: 10 :: 15 :: 25 :: 35 :: 45 :: Nil)
      .addGrid(rf.numTrees, 5 :: 10 :: 15 :: 20 :: 30 :: Nil)
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    // Set up 10-fold cross validation
    val numFolds = 10
    val crossval = new CrossValidator()
      .setEstimator(rf)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)

    val cvModel = crossval.fit(trainDF)

    // Save the workflow
    //cvModel.write.overwrite().save("model/RF_model_churn")

    val predictions = cvModel.transform(testDF)
    predictions.show(10)

    val metric = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val evaluator1 = metric.setMetricName("accuracy")
    val evaluator2 = metric.setMetricName("weightedPrecision")
    val evaluator3 = metric.setMetricName("weightedRecall")
    val evaluator4 = metric.setMetricName("f1")

    // compute the classification accuracy, precision, recall, f1 measure and error on test data.
    val accuracy = evaluator1.evaluate(predictions)
    val precision = evaluator2.evaluate(predictions)
    val recall = evaluator3.evaluate(predictions)
    val f1 = evaluator4.evaluate(predictions)

    // Print the performance metrics
    println("Accuracy = " + accuracy);
    println("Precision = " + precision)
    println("Recall = " + recall)
    println("F1 = " + f1)
    println(s"Test Error = ${1 - accuracy}")

    // Shutdown Spark cluster and H2O
    spark.stop()
  }

}
