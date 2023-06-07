import React, { useState, ChangeEvent } from "react";

interface InputBoxProps {
  setTex: React.Dispatch<React.SetStateAction<string>>;
  tex: string;
}

const InputBox: React.FC<InputBoxProps> = ({ tex, setTex }): JSX.Element => {
  const updateTextBoxContent = (text: string) => {
    setTex(text);
  };

  const handleTextBoxChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setTex(e.target.value);
  };

  return (
    <div className="border rounded-lg border-slate-800 resize-none h-1/3 w-full">
      <textarea
        placeholder="Enter TeX here..."
        value={tex}
        onChange={handleTextBoxChange}
        className="min-h-full min-w-full rounded-lg"
      ></textarea>
    </div>
  );
};

export default InputBox;
