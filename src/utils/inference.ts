import Image from "image-js";
import Matrix from "ml-matrix";
import * as onnx from "onnxruntime-web";

const getImageTensor = async (path: string | null) => {
  if (!path) return;
  const img = await Image.load(path);
  const resized = img.grey().resize({ height: 28, width: 28 });
  const normalizedImage = normalizeImage(resized);
  return imageDataToTensor(normalizedImage);
};

const imageDataToTensor = (imageMatrix: Matrix): onnx.Tensor => {
  const height = imageMatrix.rows;
  const width = imageMatrix.columns;
  const imageBuffer = new Float32Array(height * width);
  const image1d = imageMatrix.to1DArray();

  for (let i = 0; i < height * width; i++) {
    imageBuffer[i] = image1d[i];
  }

  const imageTensor = new onnx.Tensor("float32", imageBuffer, [
    1,
    1,
    height,
    width,
  ]);
  return imageTensor;
};

const normalizeImage = (image: Image) => {
  let mat: Matrix = image.getMatrix().div(255);
  const mean = mat.mean();
  const std = mat.standardDeviation();
  return mat.subtract(mean).divide(std);
};

export { getImageTensor, imageDataToTensor, normalizeImage };
