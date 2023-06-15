import Image from "image-js";
import Matrix from "ml-matrix";
import * as onnx from "onnxruntime-web";

const getImageTensor = async (path: string | null) => {
  if (!path) return;
  const img = await Image.load(path);
  const resized = img
    .grey()
    .resize({ height: 28, width: 28, interpolation: "nearestNeighbor" });
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
  let mean = mat.mean();
  let std = mat.standardDeviation();

  //sometimes image resized turns all white and everything turns into nan
  if (mean == 0) mean = 1;
  if (std == 0) std = 1;

  return mat.subtract(mean).divide(std);
};

export { getImageTensor, imageDataToTensor, normalizeImage };
