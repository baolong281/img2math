import React, { ChangeEvent, MouseEventHandler } from "react";

interface MainButtonProps {
  text: string;
  className?: string;
  onClick: MouseEventHandler<HTMLButtonElement>;
}

const MainButton: React.FC<MainButtonProps> = ({
  text,
  onClick,
  className,
}) => {
  return (
    <div className={className + " flex justify-center align-middle"}>
      <button
        onClick={onClick}
        className="bg-blue-600 rounded-full max-h-14 px-6 py-2 text-white font-bold"
      >
        {text}
      </button>
    </div>
  );
};

export default MainButton;
