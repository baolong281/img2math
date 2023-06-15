import ImageBox from "components/ImageBox";
import InputBox from "components/InputBox";
import MathBox from "components/MathBox";
import { getImageTensor } from "../utils/inference";
import React, { useEffect, useState } from "react";
import * as onnx from "onnxruntime-web";

const createSession = async () => {
  const session = await onnx.InferenceSession.create("./mnist.onnx", {
    executionProviders: ["wasm"],
  });
  return session;
};

let modelPromise = createSession();

const App = (): JSX.Element => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [tex, setTex] = useState<string>("");

  const inference = async () => {
    const model = await modelPromise;
    let input = await getImageTensor(selectedImage);
    const feeds = { "input.1": input };
    const results = await model.run(feeds);
    console.log(results["241"].data);
    return results["241"].data;
  };

  useEffect(() => {
    if (!selectedImage) return;
    const results = inference();
  }, [selectedImage]);

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
