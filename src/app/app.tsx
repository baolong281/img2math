import ImageBox from "components/ImageBox";
import InputBox from "components/InputBox";
import MathBox from "components/MathBox";
import React, { useState } from "react";

const App = (): JSX.Element => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [tex, setTex] = useState<string>("");

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
