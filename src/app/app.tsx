import ImageBox from "components/ImageBox";
import InputBox from "components/InputBox";
import MathBox from "components/MathBox";
import React, { useEffect, useState } from "react";
import * as onnx from "onnxruntime-web";
import Image from "image-js";
import Matrix from "ml-matrix";

const App = (): JSX.Element => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [tex, setTex] = useState<string>("");

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

  useEffect(() => {
    getImageTensor(selectedImage);
  }, [selectedImage]);

  useEffect(() => {
    const inference = async () => {
      const session = await onnx.InferenceSession.create("./mnist.onnx", {
        executionProviders: ["wasm"],
      });
      const input = new onnx.Tensor(
        "float32",
        new Float32Array(28 * 28),
        [1, 1, 28, 28]
      );
      const feeds = { "input.1": input };
      const results = await session.run(feeds);
      return results["241"].data;
    };
    console.log(inference());
  }, []);

  return (
    <div className="flex flex-col w-[48rem] h-[38rem] p-4 gap-4 overflow-hidden font-sans">
      <ImageBox
        selectedImage={selectedImage}
        setSelectedImage={setSelectedImage}
      />
      <InputBox tex={tex} setTex={setTex} />
      <MathBox tex={tex} />
    </div>
  );
};

export default App;
