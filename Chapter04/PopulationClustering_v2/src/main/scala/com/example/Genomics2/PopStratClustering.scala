package com.example.Genomics2

import hex.FrameSplitter
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.h2o.H2OContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.bdgenomics.adam.rdd.ADAMContext._
import org.bdgenomics.formats.avro.{ Genotype, GenotypeAllele}
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{ Vectors, Vector }
import org.apache.spark.ml.clustering.KMeans
import water.fvec.Frame
import java.io._
import org.apache.spark.SparkContext
import org.apache.spark.h2o.H2OContext
import org.apache.spark.sql.types.{ IntegerType, StringType, StructField, StructType }

import org.apache.spark.ml.feature.{ VectorAssembler, Normalizer }
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.PCA

import water.{ Job, Key }
import water.fvec.Frame

import org.apache.spark.h2o._
import java.io.File
import water._

import scala.collection.JavaConverters._
import scala.collection.immutable.Range.inclusive
import scala.io.Source

object PopStratClusterings {
  def main(args: Array[String]): Unit = {
    val genotypeFile = "C:/Users/admin-karim/Downloads/genotypes.vcf"
    val panelFile = "C:/Users/admin-karim/Downloads/genotypes.panel"

    val sparkSession: SparkSession =
      SparkSession.builder.appName("PopStrat").master("local[*]").getOrCreate()
    val sc: SparkContext = sparkSession.sparkContext

    val populations = Set("GBR", "MXL", "ASW", "CHB", "CLM")
    def extract(file: String, filter: (String, String) => Boolean): Map[String, String] = {
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
    val allGenotypes: RDD[Genotype] = sc.loadGenotypes(genotypeFile).rdd
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
    val variantsBySampleId: RDD[(String, Iterable[SampleVariant])] =
      variantsRDD.groupBy(_.sampleId)
    val sampleCount: Long = variantsBySampleId.count()
    println("Found " + sampleCount + " samples")

    val variantsByVariantId: RDD[(Int, Iterable[SampleVariant])] =
      variantsRDD.groupBy(_.variantId).filter {
        case (_, sampleVariants) => sampleVariants.size == sampleCount
      }

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

    val sortedVariantsBySampleId: RDD[(String, Array[SampleVariant])] =
      filteredVariantsBySampleId.map {
        case (sampleId, variants) =>
          (sampleId, variants.toArray.sortBy(_.variantId))
      }

    println(s"Sorted by Sample ID RDD: " + sortedVariantsBySampleId.first())

    val header = StructType(
      Array(StructField("Region", StringType)) ++
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

    //val featureVectorsRDD = rowRDD.map { x: Row => x.getAs[Vector](0) }

    // Create the SchemaRDD from the header and rows and convert the SchemaRDD into a Spark dataframe
    val sqlContext = sparkSession.sqlContext
    val schemaDF = sqlContext.createDataFrame(rowRDD, header).drop("Region")
    schemaDF.printSchema()
    schemaDF.show(10)

    val featureCols = schemaDF.columns

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val assembleDF = assembler.transform(schemaDF).select("features")
    assembleDF.show()

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(50)
      .fit(assembleDF)

    val pcaDF = pca.transform(assembleDF).select("pcaFeatures").withColumnRenamed("pcaFeatures", "features")
    pcaDF.show()

    val iterations = 20
    for (i <- 2 to iterations) {
      // Trains a k-means model.
      val kmeans = new KMeans().setK(i).setSeed(12345L)
      val model = kmeans.fit(pcaDF)

      // Evaluate clustering by computing Within Set Sum of Squared Errors.
      val WSSSE = model.computeCost(pcaDF)
      println("Within Set Sum of Squared Errors for k = " + i + " is " + WSSSE)
    }
    sparkSession.stop()
  }
}