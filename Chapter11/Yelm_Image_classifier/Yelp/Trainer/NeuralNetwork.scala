package Yelp.Trainer

import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import org.apache.commons.io.FileUtils
import java.io.{DataInputStream, DataOutputStream, FileInputStream}
import java.nio.file.{Files, Paths}

object NeuralNetwork {  
  def loadNN(NNconfig: String, NNparams: String) = {
    // get neural network config
    val confFromJson: MultiLayerConfiguration = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File(NNconfig)))    
     // get neural network parameters 
    val dis: DataInputStream = new DataInputStream(new FileInputStream(NNparams))
    val newParams = Nd4j.read(dis)    
     // creating network object
    val savedNetwork: MultiLayerNetwork = new MultiLayerNetwork(confFromJson)
    savedNetwork.init()
    savedNetwork.setParameters(newParams)    
    savedNetwork
  }
  
  def saveNN(model: MultiLayerNetwork, NNconfig: String, NNparams: String) = {
    // save neural network config
    FileUtils.write(new File(NNconfig), model.getLayerWiseConfigurations().toJson())     
    // save neural network parms
    val dos: DataOutputStream = new DataOutputStream(Files.newOutputStream(Paths.get(NNparams)))
    Nd4j.write(model.params(), dos)
  }  
}