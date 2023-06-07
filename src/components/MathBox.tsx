import React, { useState, ChangeEvent } from "react";
import MainButton from "./MainButton";
import { BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

interface MathBoxProps {
  tex: string;
}

const MathBox: React.FC<MathBoxProps> = ({ tex }): JSX.Element => {
  const handleClick = () => {};

  //<div
  //dangerouslySetInnerHTML={{ __html: doc.documentElement.outerHTML }}
  //></div>

  return (
    <div>
      <div className="flex justify-center align-middle">TeX Preview</div>
      <div className="p-4 resize-none h-1/3 w-full flex flex-col justify-center align-middle gap-4 mt-8">
        <BlockMath math={tex} />
        <MainButton text="Copy image to clipboard" onClick={handleClick} />
      </div>
    </div>
  );
};

export default MathBox;
