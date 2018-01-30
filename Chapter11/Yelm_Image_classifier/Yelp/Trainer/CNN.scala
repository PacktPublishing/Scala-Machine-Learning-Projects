package Yelp.Trainer

import Yelp.Preprocessor.makeND4jDataSets._
import Yelp.IO._
import Yelp.Preprocessor.featureAndDataAligner
import java.util.Random
import java.nio.file._
import java.io.{DataOutputStream, File}
import org.apache.commons.io.FileUtils
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, OutputLayer, SubsamplingLayer, DenseLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator

object CNN {
  def trainModel(alignedData: featureAndDataAligner, bizClass: Int = 1, saveNN: String = "") = {
    val ds = makeDataSet(alignedData, bizClass)

    println("Training getting started!..")
    println("Class for training: " + bizClass)

    val begintime = System.currentTimeMillis()
    lazy val log = LoggerFactory.getLogger(CNN.getClass)
    log.info("Training start time: " + java.util.Calendar.getInstance().getTime())
    val nfeatures = ds.getFeatures.getRow(0).length // hyper, hyper parameter

    val numRows = Math.sqrt(nfeatures).toInt // numRows * numColumns must equal columns in initial data * channels
    val numColumns = Math.sqrt(nfeatures).toInt // numRows * numColumns must equal columns in initial data * channels
    val nChannels = 1 // would be 3 if color image w R,G,B
    val outputNum = 2 // # of classes (# of columns in output)
    val iterations = 10
    val splitTrainNum = math.ceil(ds.numExamples * 0.5).toInt // 80/20 training/test split
    val seed = 12345
    val listenerFreq = 1
    val nepochs = 5

    //val nOutPar = 500 // default was 1000.  # of output nodes in first layer
    println("Number of rows: " + ds.getFeatures.size(0))
    println("Number of columns: " + ds.getFeatures.size(1))

    /**
     * Set a neural network configuration with multiple layers
     */
    log.info("Data loading....")
    ds.normalizeZeroMeanZeroUnitVariance() // this changes ds
    System.out.println("Data loaded!" + ds.labelCounts)
    
    Nd4j.shuffle(ds.getFeatureMatrix, new Random(seed), 1) // this changes ds.  Shuffles rows
    Nd4j.shuffle(ds.getLabels, new Random(seed), 1) // this changes ds.  Shuffles labels accordingly
    val trainTest: SplitTestAndTrain = ds.splitTestAndTrain(splitTrainNum, new Random(seed)) // Random Seed not needed here

    val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .miniBatch(true)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.001)
      .list(5)
      .layer(0, new ConvolutionLayer.Builder(6, 6)
        .nIn(nChannels)
        .stride(2, 2) // default stride(2,2)
        .nOut(50) // # of feature maps
        .dropOut(0.8)
        .activation("relu")
        .weightInit(WeightInit.SIZE)
        .build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
        .build())
      .layer(2, new ConvolutionLayer.Builder()
        .nOut(100)
        .dropOut(0.5)
        .weightInit(WeightInit.XAVIER)
        .activation("relu")
        .build())
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
        .build())
      .layer(4, new DenseLayer.Builder()
        .nOut(100)
        .activation("relu")
        .build())
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .nOut(outputNum)
        .weightInit(WeightInit.XAVIER)
        .activation("softmax")
        .build())
      .backprop(true).pretrain(false)

    new ConvolutionLayerSetup(builder, numRows, numColumns, nChannels)

    val conf: MultiLayerConfiguration = builder.build()

    log.info("Model building started .....")
    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)
    log.info("Model built!")

    log.info("Model training started....")
    System.out.println("Training on " + trainTest.getTrain.labelCounts())
    model.fit(trainTest.getTrain)
    log.info("Model trained!")

    log.info("Model evaluation started....")
    System.out.println("Testing on " + trainTest.getTest.labelCounts)
    val eval = new Evaluation(outputNum)
    val output: INDArray = model.output(trainTest.getTest.getFeatureMatrix) // these are the predictions
    eval.eval(trainTest.getTest.getLabels, output) // this changes the eval object (not to be confused w method w same name)
    log.info(eval.stats())

    val endtime = System.currentTimeMillis()
    log.info("End time: " + java.util.Calendar.getInstance().getTime())
    log.info("computation time: " + (endtime - begintime) / 1000.0 + " seconds")

    log.info("Write results....")

    if (!saveNN.isEmpty) {
      // model config
      FileUtils.write(new File(saveNN + ".json"), model.getLayerWiseConfigurations().toJson())
      // model parameters
      val dos: DataOutputStream = new DataOutputStream(Files.newOutputStream(Paths.get(saveNN + ".bin")))
      Nd4j.write(model.params(), dos)
    }
    log.info("****************Example finished********************")
  }
}