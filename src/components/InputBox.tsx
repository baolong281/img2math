import React, { useState, ChangeEvent } from "react";

interface InputBoxProps {}

const InputBox: React.FC<InputBoxProps> = (): JSX.Element => {
  const [textBoxContent, setTextBoxContent] = useState("");

  const updateTextBoxContent = (text: string) => {
    setTextBoxContent(text);
  };

  const handleTextBoxChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setTextBoxContent(e.target.value);
  };

  return (
    <div className="border rounded-lg border-slate-800 resize-none h-1/3 w-full">
      <textarea
        placeholder="Enter TeX here..."
        value={textBoxContent}
        onChange={handleTextBoxChange}
        className="min-h-full min-w-full rounded-lg"
      ></textarea>
    </div>
  );
};

export default InputBox;
