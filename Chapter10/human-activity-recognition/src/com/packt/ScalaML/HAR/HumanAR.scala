package com.packt.ScalaML.HAR

import ml.dmlc.mxnet.Context
import LSTMNetworkConstructor.LSTMModel
import scala.collection.mutable.ArrayBuffer
import ml.dmlc.mxnet.optimizer.Adam
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.optimizer.RMSProp
import org.sameersingh.scalaplot.MemXYSeries
import org.sameersingh.scalaplot.XYData
import org.sameersingh.scalaplot.XYChart
import org.sameersingh.scalaplot.Style._
import org.sameersingh.scalaplot.gnuplot.GnuplotPlotter
import org.sameersingh.scalaplot.jfreegraph.JFGraphPlotter

object HumanAR {  
  def main(args: Array[String]): Unit = {       
    val ctx = Context.cpu() // Retrieves the name of this Context object

    val datasetPath = "UCI_HAR_Dataset/"
    val trainDataPath = s"$datasetPath/train/Inertial Signals"
    val trainLabelPath = s"$datasetPath/train/y_train.txt"
    val testDataPath = s"$datasetPath/test/Inertial Signals"
    val testLabelPath = s"$datasetPath/test/y_test.txt"
    
    val trainData = Utils.loadData(trainDataPath, "train")
    val trainLabels = Utils.loadLabels(trainLabelPath)
    val testData = Utils.loadData(testDataPath, "test")
    val testLabels = Utils.loadLabels(testLabelPath)

    val trainingDataCount = trainData.length // 7352 training series (with 50% overlap between each series)
    val testDataCount = testData.length // 2947 testing series
    val nSteps = trainData(0).length // 128 timesteps per series
    val nInput = trainData(0)(0).length // 9 input parameters per timestep

    // LSTM Neural Network's internal structure
    val nHidden = 28 // Hidden layer num of features
    val nClasses = 6 // Total classes (should go up, or should go down)

    // Hyper parameters for the LSTM training 
    val learningRate = 0.001f
    val trainingIters = trainingDataCount * 100  // Loop 100 times on the dataset, for a total of 7352000 iterations
    val batchSize = 1500
    val displayIter = 15000  // To show test set accuracy during training
    val numLstmLayer = 1

    val model = LSTMNetworkConstructor.setupModel(nSteps, nInput, nHidden, nClasses, batchSize, ctx = ctx)

    val opt = new RMSProp(learningRate = learningRate)

    val paramBlocks = model.symbol.listArguments()
      .filter(x => x != "data" && x != "softmax_label")
      .zipWithIndex.map { case (name, idx) =>
        val state = opt.createState(idx, model.argsDict(name))
        (idx, model.argsDict(name), model.gradDict(name), state, name)
      }.toArray

    // To keep track of training's performance
    val testLosses = ArrayBuffer[Float]()
    val testAccuracies = ArrayBuffer[Float]()
    val trainLosses = ArrayBuffer[Float]()
    val trainAccuracies = ArrayBuffer[Float]()
    
    // Perform Training steps with "batch_size" iterations at each loop
    var step = 1
    while (step * batchSize <= trainingIters) {
      val (batchTrainData, batchTrainLabel) = {
        val idx = ((step - 1) * batchSize) % trainingDataCount
        if (idx + batchSize <= trainingDataCount) {
          val datas = trainData.drop(idx).take(batchSize)
          val labels = trainLabels.drop(idx).take(batchSize)
          (datas, labels)
        } else {
          val right = (idx + batchSize) - trainingDataCount
          val left = trainingDataCount - idx
          val datas = trainData.drop(idx).take(left) ++ trainData.take(right)
          val labels = trainLabels.drop(idx).take(left) ++ trainLabels.take(right)
          (datas, labels)
        }
      }
      
      model.data.set(batchTrainData.flatten.flatten)
      model.label.set(batchTrainLabel)
      
      model.exec.forward(isTrain = true)
      model.exec.backward()

      paramBlocks.foreach { case (idx, weight, grad, state, name) =>
        opt.update(idx, weight, grad, state)
      }
      val (acc, loss) = getAccAndLoss(model.exec.outputs(0), batchTrainLabel)

      trainLosses += loss / batchSize
      trainAccuracies += acc / batchSize

      // Evaluate network only at some steps for faster training: 
      if ( (step * batchSize % displayIter == 0) || (step == 1) || (step * batchSize > trainingIters) ) {
        println(s"Iter ${step * batchSize}, Batch Loss = ${"%.6f".format(loss / batchSize)}, Accuracy = ${acc / batchSize}")


        // Evaluation on the test set
        val (testLoss, testAcc) = test(testDataCount, batchSize, testData, testLabels, model)
        
        println(s"TEST SET DISPLAY STEP:  Batch Loss = ${"%.6f".format(testLoss)}, Accuracy = $testAcc")
        testAccuracies += testAcc
        testLosses += testLoss
      }
      step += 1
    }
    
    val (finalLoss, accuracy) = test(testDataCount, batchSize, testData, testLabels, model)

    println(s"FINAL RESULT: Batch Loss= $finalLoss, Accuracy= $accuracy")
    
    model.exec.dispose()

    // visualize
    val xTrain = (0 until trainLosses.length * batchSize by batchSize).toArray.map(_.toDouble)
    val yTrainL = trainLosses.toArray.map(_.toDouble)
    val yTrainA = trainAccuracies.toArray.map(_.toDouble)
    
    val xTest = (0 until testLosses.length * displayIter by displayIter).toArray.map(_.toDouble)
    val yTestL = testLosses.toArray.map(_.toDouble)
    val yTestA = testAccuracies.toArray.map(_.toDouble)

    var series = new MemXYSeries(xTrain, yTrainL, "Train losses")
    val data = new XYData(series)
      
    series = new MemXYSeries(xTrain, yTrainA, "Train accuracies")
    data += series

    series = new MemXYSeries(xTest, yTestL, "Test losses")
    data += series
    
    series = new MemXYSeries(xTest, yTestA, "Test accuracies")
    data += series

    val chart = new XYChart("Training session's progress over iterations!", data)
    chart.showLegend = true
    val plotter = new JFGraphPlotter(chart)
    plotter.gui()
  }

  def test(testDataCount: Int, batchSize: Int, testDatas: Array[Array[Array[Float]]],
      testLabels: Array[Float], model: LSTMModel): (Float, Float) = {
    var testLoss, testAcc = 0f
    for (begin <- 0 until testDataCount by batchSize) {
      val (testData, testLabel, dropNum) = {
        if (begin + batchSize <= testDataCount) {
          val datas = testDatas.drop(begin).take(batchSize)
          val labels = testLabels.drop(begin).take(batchSize)
          (datas, labels, 0)
        } else {
          val right = (begin + batchSize) - testDataCount
          val left = testDataCount - begin
          val datas = testDatas.drop(begin).take(left) ++ testDatas.take(right)
          val labels = testLabels.drop(begin).take(left) ++ testLabels.take(right)
          (datas, labels, right)
        }
      }
      
      model.data.set(testData.flatten.flatten)
      model.label.set(testLabel)
    
      model.exec.forward(isTrain = false)
      val (acc, loss) = getAccAndLoss(model.exec.outputs(0), testLabel)
      testLoss += loss
      testAcc += acc
    }
    (testLoss / testDataCount, testAcc / testDataCount)
  }

  def getAccAndLoss(pred: NDArray, label: Array[Float], dropNum: Int = 0): (Float, Float) = {
    val shape = pred.shape
    val maxIdx = NDArray.argmax_channel(pred).toArray
    val acc = {
      val sum = maxIdx.drop(dropNum).zip(label.drop(dropNum)).foldLeft(0f){ case (acc, elem) => 
        if (elem._1 == elem._2) acc + 1 else acc
      }
      sum
    }
    val loss = pred.toArray.grouped(shape(1)).drop(dropNum).zipWithIndex.map { case (array, idx) =>
        array(maxIdx(idx).toInt)  
      }.map(-Math.log(_)).sum.toFloat

    (acc, loss)
  }
}