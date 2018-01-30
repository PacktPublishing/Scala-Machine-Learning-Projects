package Yelp.Preprocessor

import java.io.File
import javax.imageio.ImageIO
import java.awt.Color

object GrayscaleConverter {
  def main(args: Array[String]): Unit = {
    def pixels2Gray(R: Int, G: Int, B: Int): Int = (R + G + B) / 3

    def makeGray(testImage: java.awt.image.BufferedImage): java.awt.image.BufferedImage = {
      val w = testImage.getWidth
      val h = testImage.getHeight
      for {
        w1 <- (0 until w).toVector
        h1 <- (0 until h).toVector
      } yield {
        val col = testImage.getRGB(w1, h1)
        val R = (col & 0xff0000) / 65536
        val G = (col & 0xff00) / 256
        val B = (col & 0xff)
        val graycol = pixels2Gray(R, G, B)
        testImage.setRGB(w1, h1, new Color(graycol, graycol, graycol).getRGB)
      }
      testImage
    }

    val testImage = ImageIO.read(new File("data/images/preprocessed/147square.jpg"))
    val grayImage = makeGray(testImage)
    ImageIO.write(grayImage, "jpg", new File("data/images/preprocessed/147gray.jpg"))
  }
}