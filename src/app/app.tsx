import ImageBox from "components/ImageBox";
import InputBox from "components/InputBox";
import React, { useState } from "react";

const App = (): JSX.Element => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  return (
    <div className="flex flex-col w-[48rem] h-[38rem] p-4 gap-2 overflow-hidden">
      <ImageBox
        selectedImage={selectedImage}
        setSelectedImage={setSelectedImage}
      />
      <InputBox />
    </div>
  );
};

export default App;
