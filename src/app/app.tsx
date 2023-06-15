import ImageBox from "components/ImageBox";
import InputBox from "components/InputBox";
import MathBox from "components/MathBox";
import React, { useEffect, useState } from "react";
import * as onnx from "onnxruntime-web";
import Image from "image-js";

const App = (): JSX.Element => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [tex, setTex] = useState<string>("");

  const getImageTensor = async (path: string | null) => {
    if (!path) return;
    const image = await Image.load(path);
    console.log(image);
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
