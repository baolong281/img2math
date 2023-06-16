import React, { useState, ChangeEvent, useEffect } from "react";

interface InputBoxProps {
  setTex: React.Dispatch<React.SetStateAction<string>>;
  tex: string;
  setTextBox: React.Dispatch<React.SetStateAction<string>>;
  textBox: string;
}

const InputBox: React.FC<InputBoxProps> = ({
  tex,
  setTex,
  textBox,
  setTextBox,
}): JSX.Element => {
  useEffect(() => {
    setTex(convertToKatex(textBox));
    console.log(tex);
  }, [textBox]);

  const updateTeX = (text: string) => {
    setTex(text);
  };

  const handleTextBoxChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setTextBox(e.target.value);
  };

  function convertToKatex(latexString: string): string {
    //remove special only???
    let kaTeXString = latexString.replace(/\\sp/g, "^");

    return kaTeXString;
  }

  return (
    <div className="border rounded-lg border-slate-800 resize-none h-1/3 w-full">
      <textarea
        placeholder="Enter TeX here..."
        value={textBox}
        onChange={handleTextBoxChange}
        className="min-h-full min-w-full rounded-lg"
      ></textarea>
    </div>
  );
};

export default InputBox;
