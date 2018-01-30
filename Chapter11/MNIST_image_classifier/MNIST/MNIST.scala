package Yelp.MNIST

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.Logger
import org.slf4j.LoggerFactory

object MNIST {
  val log: Logger = LoggerFactory.getLogger(MNIST.getClass)

  def main(args: Array[String]): Unit = {
    val nChannels = 1 // for grayscale image
    val outputNum = 10 // number of class
    val nEpochs = 10 // number of epoch
    val iterations = 1 // number of iteration
    val seed = 12345 // Random seed for reproducibility
    val batchSize = 64 // number of batches to be sent

    log.info("Load data....")
    val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, 12345)
    val mnistTest: DataSetIterator = new MnistDataSetIterator(batchSize, false, 12345)

    log.info("Network layer construction started...")
    //First convolution layer with ReLU as activation function
    val layer_0 = new ConvolutionLayer.Builder(5, 5)
      .nIn(nChannels)
      .stride(1, 1)
      .nOut(20)
      .activation("relu")
      .build()
    
    //First subsampling layer
    val layer_1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(2, 2)
      .stride(2, 2)
      .build()

    //Second convolution layer with ReLU as activation function
    val layer_2 = new ConvolutionLayer.Builder(5, 5)
      .nIn(nChannels)
      .stride(1, 1)
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
    val layer_5 = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
      .nOut(outputNum)
      .activation("softmax")
      .build()

    log.info("Model building started...")
    val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .regularization(true).l2(0.0005)
      .learningRate(0.01)
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .list()
          .layer(0, layer_0)
          .layer(1, layer_1)
          .layer(2, layer_2)
          .layer(3, layer_3)
          .layer(4, layer_4)
          .layer(5, layer_5)
      .backprop(true).pretrain(false) // feedforward so no backprop

    // Setting up all the convlutional layers and initilalize the network
    new ConvolutionLayerSetup(builder, 28, 28, 1) //image size is 28*28
    val conf: MultiLayerConfiguration = builder.build()
    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()

    log.info("Model training started...")
    model.setListeners(new ScoreIterationListener(1))
    var i = 0
    while (i <= nEpochs) {
      model.fit(mnistTrain);
      log.info("*** Completed epoch {} ***", i)
      i = i + 1
      var ds: DataSet = null
      var output: INDArray = null

      log.info("Model evaluation....")
      val eval: Evaluation = new Evaluation(outputNum)
      while (mnistTest.hasNext()) {
        ds = mnistTest.next()
        output = model.output(ds.getFeatureMatrix(), false)
      }
      eval.eval(ds.getLabels(), output)
      println("Accuracy: " + eval.accuracy())
      println("F1 measure: " + eval.f1())
      println("Precision: " + eval.precision())
      println("Recall: " + eval.recall())

      println(eval.confusionToString())
      log.info(eval.stats())
      mnistTest.reset()
    }
    log.info("****************Example finished********************")
  }
}
