import React, { useState } from "react";
import { BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

interface MathBoxProps {
  tex: string;
}

const MathBox: React.FC<MathBoxProps> = ({ tex }): JSX.Element => {
  return (
    <div className="">
      <div className="flex font-bold justify-center align-middle">
        TeX Preview
      </div>
      <div className="mb-6 text-2xl h-3/4 gap-6 flex flex-col justify-center align-middle">
        <BlockMath math={tex} />
      </div>
    </div>
  );
};

export default MathBox;
