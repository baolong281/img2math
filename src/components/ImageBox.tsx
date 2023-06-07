import React, { useEffect, useState, ClipboardEvent, ChangeEvent } from "react";
import MainButton from "./MainButton";

interface ImageBoxProps {
  setSelectedImage: React.Dispatch<React.SetStateAction<string | null>>;
  selectedImage: string | null;
}

//box to upload images
const ImageBox: React.FC<ImageBoxProps> = ({
  setSelectedImage,
  selectedImage,
}): JSX.Element => {
  const hiddenFileInput = React.useRef<HTMLInputElement>(null);

  const handlePasteImage = (event: ClipboardEvent) => {
    if (!event.clipboardData) return; // If clipboard data is valid

    const pasteEvent = event.clipboardData;
    const file = Array.from(pasteEvent.items).find(
      (item) => item.type.indexOf("image") !== -1
    );

    if (file) {
      const fileBlob = file.getAsFile();
      if (fileBlob) {
        const imageURL = URL.createObjectURL(fileBlob);
        setSelectedImage(imageURL);
      }
    }
  };

  const handleUploadImage = (event: ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files?.[0]) return; // If file is valid

    const file = event.target.files?.[0];
    const reader = new FileReader();

    reader.onloadend = () => {
      setSelectedImage(reader.result as string);
    };

    reader.readAsDataURL(file);
  };

  const handleClick = () => {
    hiddenFileInput.current?.click();
  };

  const clearImage = () => {
    setSelectedImage(null);
  };

  return (
    <div
      className="border-2 border-slate-800 border-c rounded-lg flex justify-center p-2 h-1/3 w-full font-bold"
      onPaste={handlePasteImage}
    >
      {selectedImage ? (
        <div className="">
          <MainButton
            text="Clear image"
            onClick={clearImage}
            className="p-2 absolute top-2 left-2 m-6"
          />
          <img
            src={selectedImage}
            alt="Pasted"
            style={{ maxWidth: "100%", maxHeight: "100%" }}
          />
        </div>
      ) : (
        <div className=" grid grid-cols-1 align-middle max mt-8">
          <input
            type="file"
            accept="image/"
            onChange={handleUploadImage}
            ref={hiddenFileInput}
            style={{ display: "none" }}
          />
          <MainButton
            text="Upload Image"
            onClick={handleClick}
            className="text-2xl"
          />
          <div className="flex align-middle justify-center font-semibold">
            Or drop / paste
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageBox;
