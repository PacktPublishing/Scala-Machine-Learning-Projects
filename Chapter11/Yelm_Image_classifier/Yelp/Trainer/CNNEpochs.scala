package Yelp.Trainer

import Yelp.Preprocessor.makeND4jDataSets.makeDataSetTE
import Yelp.Preprocessor.featureAndDataAligner
import Yelp.Preprocessor.makeND4jDataSets._
import java.util.Random
import java.nio.file._
import java.io.{ DataOutputStream, File }
import org.apache.commons.io.FileUtils
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ ConvolutionLayer, OutputLayer, SubsamplingLayer, DenseLayer }
import org.deeplearning4j.nn.conf.{ MultiLayerConfiguration, NeuralNetConfiguration }
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

object CNNEpochs {
  def trainModelEpochs(alignedData: featureAndDataAligner, businessClass: Int = 1, saveNN: String = "") = {
    val ds = makeDataSet(alignedData, businessClass)
    println("commence training!!")
    println("class for training: " + businessClass)
    val begintime = System.currentTimeMillis()
    lazy val log = LoggerFactory.getLogger(CNN.getClass)
    log.info("Begin time: " + java.util.Calendar.getInstance().getTime())

    val nfeatures = ds.getFeatures.getRow(0).length // hyper, hyper parameter
    val numRows = Math.sqrt(nfeatures).toInt // numRows * numColumns must equal columns in initial data * channels
    val numColumns = Math.sqrt(nfeatures).toInt // numRows * numColumns must equal columns in initial data * channels
    val nChannels = 1 // would be 3 if color image w R,G,B
    val outputNum = 2// # of classes (# of columns in output)
    val iterations = 1
    val splitTrainNum = math.ceil(ds.numExamples * 0.75).toInt // 80/20 training/test split
    val seed = 12345
    val listenerFreq = 1
    val nepochs = 2
    val nbatch = 128 // recommended between 16 and 128

    //val nOutPar = 500 // default was 1000.  # of output nodes in first layer  
    println("rows: " + ds.getFeatures.size(0))
    println("columns: " + ds.getFeatures.size(1))

    /**
     * Set a neural network configuration with multiple layers
     */
    log.info("Load data....")
    ds.normalizeZeroMeanZeroUnitVariance() //  A data transform (example/outcome pairs). The outcomes are specifically for neural network encoding such that any labels that are considered true are 1s. The rest are zeros.
    System.out.println("Loaded " + ds.labelCounts)
    Nd4j.shuffle(ds.getFeatureMatrix, new Random(seed), 1) // this shuffles rows in the ds.  
    Nd4j.shuffle(ds.getLabels, new Random(seed), 1) // this shuffles the labels accordingly
    val trainTest: SplitTestAndTrain = ds.splitTestAndTrain(splitTrainNum, new Random(seed)) // Random Seed not needed here      

    // creating epoch dataset iterator
    val dsiterTr = new ListDataSetIterator(trainTest.getTrain.asList(), nbatch)
    val dsiterTe = new ListDataSetIterator(trainTest.getTest.asList(), nbatch)
    val epochitTr: MultipleEpochsIterator = new MultipleEpochsIterator(nepochs, dsiterTr)
    val epochitTe: MultipleEpochsIterator = new MultipleEpochsIterator(nepochs, dsiterTe)

    //First convolution layer with ReLU as activation function
    val layer_0 = new ConvolutionLayer.Builder(6, 6)
        .nIn(nChannels)
        .stride(2, 2) // default stride(2,2)
        .nOut(20) // # of feature maps
        .dropOut(0.5)
        .activation("relu") // rectified linear units
        .weightInit(WeightInit.RELU)
        .build()
    
    //First subsampling layer
    val layer_1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(2, 2)
      .stride(2, 2)
      .build()

    //Second convolution layer with ReLU as activation function
    val layer_2 = new ConvolutionLayer.Builder(6, 6)
      .nIn(nChannels)
      .stride(2, 2)
      .nOut(50)
      .activation("relu")
      .build()

    //Second subsampling layer
    val layer_3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(2, 2)
      .stride(2, 2)
      .build()

    //Dense layer
    val layer_4 = new DenseLayer.Builder()
      .activation("relu")
      .nOut(500)
      .build()

    // Final and fully connected layer with Softmax as activation function
    val layer_5 = new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
      .nOut(outputNum)
      .weightInit(WeightInit.XAVIER)
      .activation("softmax")
      .build()
      
      val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .miniBatch(true)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .regularization(true).l2(0.0005)
      .learningRate(0.01)
      .list(6)
          .layer(0, layer_0)
          .layer(1, layer_1)
          .layer(2, layer_2)
          .layer(3, layer_3)
          .layer(4, layer_4)
          .layer(5, layer_5)
      .backprop(true).pretrain(false)
      
    /*
    //Configuration for a multilayer network:3 layer network. In fact 3 layers network. 1st layer is a convlutional layer with ReLu as activation, 
    //2nd layer is a subsampling layer witth no activation required, layer 3 is a dense layer with activation ReLU
    //The 4th layer is the final output layer with activation Softmax.
    //We are using SGD as the optimization algoritm, with an LR of 0.01 and a momentum of 0.9
    val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .miniBatch(true)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.01)
      .momentum(0.9)
      .list(4)
      .layer(0, new ConvolutionLayer.Builder(6, 6)
        .nIn(nChannels)
        .stride(2, 2) // default stride(2,2)
        .nOut(20) // # of feature maps
        .dropOut(0.5)
        .activation("relu") // rectified linear units
        .weightInit(WeightInit.RELU)
        .build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
        .build())
      .layer(2, new DenseLayer.Builder()
        .nOut(40)
        .activation("relu")
        .build())
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .nOut(outputNum)
        .weightInit(WeightInit.XAVIER)
        .activation("softmax")
        .build())
      .backprop(true).pretrain(false)
      */
      
    new ConvolutionLayerSetup(builder, numRows, numColumns, nChannels)
    val conf: MultiLayerConfiguration = builder.build()

    log.info("Build model....")
    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)

    log.info("Train model....")
    System.out.println("Training on " + dsiterTr.getLabels) // this might return null
    model.fit(epochitTr)

    // I think this could be done without an iterator and batches.
    log.info("Evaluate model....")
    System.out.println("Testing on ...")
    val eval = new Evaluation(outputNum)
    while (epochitTe.hasNext) {
      val testDS = epochitTe.next(nbatch)
      val output: INDArray = model.output(testDS.getFeatureMatrix)
      eval.eval(testDS.getLabels(), output)
    }
    System.out.println(eval.stats())

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
