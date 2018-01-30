package com.example.Genomics2

import java.io._

import hex.FrameSplitter
import hex.deeplearning.DeepLearning
import hex.deeplearning.DeepLearningModel.DeepLearningParameters
import hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation
import org.apache.spark.SparkContext
import org.apache.spark.h2o.H2OContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

import org.bdgenomics.adam.rdd.ADAMContext._
import org.bdgenomics.formats.avro.{ Genotype, GenotypeAllele }
import water.{ Job, Key }
import water.support.ModelMetricsSupport
import water.fvec.Frame

import org.apache.spark.h2o._
import java.io.File

import _root_.hex.{ModelMetrics, ModelMetricsSupervised, ModelMetricsMultinomial}

import scala.collection.JavaConverters._
import scala.collection.immutable.Range.inclusive
import scala.io.Source

object PopStratClassification {
  def main(args: Array[String]): Unit = {
    val genotypeFile = "C:/Users/admin-karim/Downloads/genotypes.vcf"
    val panelFile = "C:/Users/admin-karim/Downloads/genotypes.panel"

    val sparkSession: SparkSession =
      SparkSession.builder.appName("PopStrat").master("local[*]").getOrCreate()
    val sc: SparkContext = sparkSession.sparkContext

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

    val writer_0 = new PrintWriter(new File("output_1.txt"))
    writer_0.write("Found " + sampleCount + " samples")
    //writer.write(s"Confusion Matrix"+ cm)
    //writer.write("Prediction Matrix"+ result)
    writer_0.close()

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
    val sqlContext = sparkSession.sqlContext
    val schemaDF = sqlContext.createDataFrame(rowRDD, header)
    val h2oContext = H2OContext.getOrCreate(sparkSession)
    import h2oContext.implicits._
        
    val dataFrame = h2oContext.asH2OFrame(schemaDF)
    dataFrame
      .replace(dataFrame.find("Region"),
        dataFrame.vec("Region").toCategoricalVec())
      .remove()
    dataFrame.update()

    // Split the dataframe into 60% training, 20% test, and 20% validation data
    val frameSplitter = new FrameSplitter(
      dataFrame,
      Array(.8, .1),
      Array("training", "test", "validation").map(Key.make[Frame]),
      null)

    water.H2O.submitTask(frameSplitter)
    val splits = frameSplitter.getResult
    val training = splits(0)
    val test = splits(1)
    val validation = splits(2)

    // Set the parameters for our deep learning model.
    val deepLearningParameters = new DeepLearningParameters()
    deepLearningParameters._train = training
    deepLearningParameters._valid = validation
    deepLearningParameters._response_column = "Region"
    deepLearningParameters._epochs = 100
    deepLearningParameters._l2 = 0.01
    deepLearningParameters._seed = 1234567
    deepLearningParameters._activation = Activation.RectifierWithDropout
    deepLearningParameters._hidden = Array[Int](32, 64, 128)

    // Train the deep learning model
    val deepLearning = new DeepLearning(deepLearningParameters)
    val deepLearningTrained = deepLearning.trainModel
    val trainedModel = deepLearningTrained.get

    val error = trainedModel.classification_error()
    println("Training Error: " + error)

    val predict = trainedModel.score(test)('predict)
    trainedModel.score(test)('predict)
    
    /*
    val h2oContext2 = H2OContext.getOrCreate(sc)
    import h2oContext2._
    import h2oContext2.implicits._
    
    val predictionsFromModel = asRDD[DoubleHolder](predict).collect.map(_.result.getOrElse(Double.NaN))
    predictionsFromModel.foreach{ value => println(value)}
    * 
    */

    // Collect model metrics and evaluate model quality
    val trainMetrics = ModelMetricsSupport.modelMetrics[ModelMetricsMultinomial](trainedModel, test)
    val met = trainMetrics.cm()
    println("Accuracy: "+ met.accuracy())
    println("MSE: "+ trainMetrics.mse)
    println("RMSE: "+ trainMetrics.rmse)
    println("R2: " + trainMetrics.r2)
     
    // Shutdown Spark cluster and H2O
    h2oContext.stop(stopSparkContext = true)
    sparkSession.stop()
  }
}
