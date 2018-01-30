package com.packt.ScalaML.HAR

import ml.dmlc.mxnet.Executor
import ml.dmlc.mxnet.NDArray
import ml.dmlc.mxnet.Symbol
import ml.dmlc.mxnet.Shape
import ml.dmlc.mxnet.Context
import ml.dmlc.mxnet.Uniform

object LSTMNetworkConstructor {case class LSTMModel(exec: Executor, symbol: Symbol, data: NDArray, label: NDArray,
      argsDict: Map[String, NDArray], gradDict: Map[String, NDArray])

  final case class LSTMState(c: Symbol, h: Symbol)
  final case class LSTMParam(i2hWeight: Symbol, i2hBias: Symbol,
                                                         h2hWeight: Symbol, h2hBias: Symbol)

  // LSTM Cell symbol
  private def lstmCell(
    numHidden: Int,
    inData: Symbol,
    prevState: LSTMState,
    param: LSTMParam,
    seqIdx: Int,
    layerIdx: Int,
    dropout: Float = 0f): LSTMState = {
    val inDataa = {
      if (dropout > 0f) Symbol.Dropout()()(Map("data" -> inData, "p" -> dropout))
      else inData
    }
    // add an hidden layer of size numHidden * 4 (numHidden set to 28) that taks as input, the inputdata     
    val i2h = Symbol.FullyConnected(s"t${seqIdx}_l${layerIdx}_i2h")()(Map("data" -> inDataa,
                                                       "weight" -> param.i2hWeight,
                                                       "bias" -> param.i2hBias,
                                                       "num_hidden" -> numHidden * 4))
        // add an hidden layer of size numHidden * 4 (numHidden set to 28) that taks as input, the the previous output of the cell 
    val h2h = Symbol.FullyConnected(s"t${seqIdx}_l${layerIdx}_h2h")()(Map("data" -> prevState.h,
                                                       "weight" -> param.h2hWeight,
                                                       "bias" -> param.h2hBias,
                                                       "num_hidden" -> numHidden * 4))
    //concatenate them                                       
    val gates = i2h + h2h
    
    //make 4 copies of gates
    val sliceGates = Symbol.SliceChannel(s"t${seqIdx}_l${layerIdx}_slice")(gates)(
        Map("num_outputs" -> 4))
    // compute the gates
    val ingate = Symbol.Activation()()(Map("data" -> sliceGates.get(0), "act_type" -> "sigmoid"))
    val inTransform = Symbol.Activation()()(Map("data" -> sliceGates.get(1), "act_type" -> "tanh"))
    val forgetGate = Symbol.Activation()()(Map("data" -> sliceGates.get(2), "act_type" -> "sigmoid"))
    val outGate = Symbol.Activation()()(Map("data" -> sliceGates.get(3), "act_type" -> "sigmoid"))
    // get the new cell state and the output
    val nextC = (forgetGate * prevState.c) + (ingate * inTransform)
    val nextH = outGate * Symbol.Activation()()(Map("data" -> nextC, "act_type" -> "tanh"))
    LSTMState(c = nextC, h = nextH)
  }

  //gives a symbolic model for the deep NN
  private def getSymbol(seqLen: Int, numHidden: Int, numLabel: Int,
      numLstmLayer: Int = 1, dropout: Float = 0f): Symbol = {
	  //symbolic training and label variables
    var inputX = Symbol.Variable("data")
    val inputY = Symbol.Variable("softmax_label")

	//the initial parametrs and cells
    var paramCells = Array[LSTMParam]()
    var lastStates = Array[LSTMState]()
	//numLstmLayer is 1 
    for (i <- 0 until numLstmLayer) {
      paramCells = paramCells :+ LSTMParam(i2hWeight = Symbol.Variable(s"l${i}_i2h_weight"),
                                                       i2hBias = Symbol.Variable(s"l${i}_i2h_bias"),
                                                       h2hWeight = Symbol.Variable(s"l${i}_h2h_weight"),
                                                       h2hBias = Symbol.Variable(s"l${i}_h2h_bias"))
      lastStates = lastStates :+ LSTMState(c = Symbol.Variable(s"l${i}_init_c"),
                                                                      h = Symbol.Variable(s"l${i}_init_h"))
    }
    assert(lastStates.length == numLstmLayer)
    
	//transform the input shape from the Shape(1500,128,9) to 128 inputs of Shape(1500,9)
	//if you are not familiar with this notion take the example of transforming an array of dim (2,3) 
	// to three arrays of dim(2) [ [1,2,3],[4,5,6] ] => [1,4] [2,5] [3,6]
	// 1500 is the batch size, 128 is the lengh of the row in the dataset, and 9 are the reading from 9 sensors
	// after this transformation, we get 128 inputs of the size 1500 where each row is 
	// [body_acc_x,body_acc_y,body_acc_z,body_gyro_x,body_gyro_y,body_gyro_z,total_acc_x,total_acc_y,total_acc_z]
    val lstmInputs = Symbol.SliceChannel()(inputX)(Map("axis" -> 1, "num_outputs" -> seqLen, "squeeze_axis" -> 1))

    var hiddenAll = Array[Symbol]()
    var dpRatio = 0f
    var hidden: Symbol = null
	
	//for each one of the 128 inputs, creat a LSTM Cell
    for (seqIdx <- 0 until seqLen) {
      hidden = lstmInputs.get(seqIdx)
      // stack LSTM
	  //numLstmLayer is 1 so the loop will be executed only one time
      for (i <- 0 until numLstmLayer) {
        if (i == 0) dpRatio = 0f else dpRatio = dropout
		//for each one of the 128 inputs, creat a LSTM Cell
        val nextState = lstmCell(numHidden, inData = hidden,
                                prevState = lastStates(i),
                                param = paramCells(i),
                                seqIdx = seqIdx, layerIdx = i, dropout = dpRatio)
        hidden = nextState.h // has no effect
        lastStates(i) = nextState // has no effect
      }
      //  add dropout before softmax, has no effect since drpout is 0 due to numLstmLayer == 1
      if (dropout > 0f) hidden = Symbol.Dropout()()(Map("data" -> hidden, "p" -> dropout))
      // store the lstm cells output layers
      hiddenAll = hiddenAll :+ hidden
    }

	// val finalOut = hiddenAll(hiddenAll.length - 1)
    //concatenate cell's outputs
    val finalOut = hiddenAll.reduce(_+_);
	//connect them to an output layer that corresponds to the 6 label
   // WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,  SITTING, STANDING, LAYING.
    val fc = Symbol.FullyConnected()()(Map("data" -> finalOut, "num_hidden" -> numLabel))
	//softmax activation against the label
    Symbol.SoftmaxOutput()()(Map("data" -> fc, "label" -> inputY))
  }

  def setupModel(seqLen: Int, nInput: Int, numHidden: Int, numLabel: Int, batchSize: Int,
      numLstmLayer: Int = 1, dropout: Float = 0f, ctx: Context = Context.cpu()): LSTMModel = {
	  //get the symbolic model
    val sym = LSTMNetworkConstructor.getSymbol(seqLen, numHidden, numLabel, numLstmLayer = numLstmLayer)
    val argNames = sym.listArguments()
    val auxNames = sym.listAuxiliaryStates()
	// defining the initial argument and binding them to the model
    val initC = for (l <- 0 until numLstmLayer) yield (s"l${l}_init_c", (batchSize, numHidden))
    val initH = for (l <- 0 until numLstmLayer) yield (s"l${l}_init_h", (batchSize, numHidden))
    val initStates = (initC ++ initH).map(x => x._1 -> Shape(x._2._1, x._2._2)).toMap

    val dataShapes = Map("data" -> Shape(batchSize, seqLen, nInput)) ++ initStates

    val (argShapes, outShapes, auxShapes) = sym.inferShape(dataShapes)

    val initializer = new Uniform(0.1f)
    val argsDict = argNames.zip(argShapes).map { case (name, shape) =>
       val nda = NDArray.zeros(shape, ctx)
       if (!dataShapes.contains(name) && name != "softmax_label") {
         initializer(name, nda)
       }
       name -> nda
    }.toMap

    val argsGradDict = argNames.zip(argShapes)
                                           .filter(x => x._1 != "softmax_label" && x._1 != "data")
                                           .map( x => x._1 -> NDArray.zeros(x._2, ctx) ).toMap

    val auxDict = auxNames.zip(auxShapes.map(NDArray.zeros(_, ctx))).toMap
    val exec = sym.bind(ctx, argsDict, argsGradDict, "write", auxDict, null, null)

    val data = argsDict("data")
    val label = argsDict("softmax_label")
    LSTMModel(exec, sym, data, label, argsDict, argsGradDict)
  }
}